"""Power berry calendar: which ROM berry source this morning's plan spends a trip on.

Every berry is +10 max stamina (the ending's stamina term pays ``(max_stamina - 100) >> 1``, 5 points a berry, 50 at
ten), +10 happiness, and a bigger watering bar from the day it is eaten, so the cheap early sources come first.

Two kinds of source:

* **Opportunistic** - the hoe (two berries, 1/64 a swing), the mountain stump (1/16 a six-chop tree) and the house
  berry armed by the Egg Festival prize. The ROM sets the permanent bit when the berry *spawns*, so one that is not
  picked up is gone for good: e2e ``s4`` ended with ``$7F1F64`` bits 0x0800, 0x1000 and 0x0400 all set and
  ``power_berry_count`` 0, three berries lost. ``tasks.power_berry.collect_spawned_berry`` (called from the day plan's
  step and from the hoeing tasks) picks those up. A scheduled stump trip covers runs without wood-gathering trips.
* **Scheduled** - this module. One trip a day at most, decided from RAM alone, each gated by the permanent receipt
  flag the ROM sets (all verified to survive the night reset, ROM receipts in ``recordings/berries``):

| source           | receipt flag (clear = still to get) | window                                              | measured cost |
| ---------------- | ----------------------------------- | --------------------------------------------------- | ------------- |
| statue           | ``$7F1F64 & 0x0200``                | any clear day from Y1 Spring D4, hammer (1/4 a swing) | ~1 game hour  |
| stump            | ``$7F1F64 & 0x0400``                | clear morning after the four early berries, if still missing | ~4 game hours |
| fish             | ``$7F1F66 & 0x2000``                | any clear day but Winter, catch a fish and put it back | ~4 game hours |
| cave             | ``$7F1F6A & 0x0040``                | any season once ``$7F1F64 & 0x0002`` (first snowstorm, by Y1 Winter D8) | ~2 game hours |
| chicken_trade    | ``$7F1F66 & 0x4000``                | Fall, shop open, dry, two or more adult hens        | ~5 game hours, costs the hen |
| harvest_festival | ``$7F1F6C & 0x8000``                | Fall D12, a fish offered at the festival            | ~1 game hour  |
| egg_festival     | ``$7F1F6C & 0x3000`` (both bits)    | Fall D20 of Year 1 and 2; win the hunt              | ~10,400 frames |

The Egg Festival prize is a coin flip between the turtle shell (bit 0x1000, +21 ranch points) and the house berry's arm
(bit 0x2000). The ROM gives the other one on the next win, so Fall D20 of Year 1 and of Year 2 cover both.
The snow-flower seed is bought on Spring D23 (purchase receipt ``$7F1F6E & 0x0800``), then planted on the Winter summit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from harvest.planner.day_phase_types import PhaseSpec

BERRY_TARGET = 10
SEASON_SPRING, SEASON_SUMMER, SEASON_FALL, SEASON_WINTER = 0, 1, 2, 3
ENV_VAR = "HARVEST_POWER_BERRIES"          # 1/0 forces the calendar on or off for an A/B, whatever the strategy

# WRAM words, as ``read_ram_u16`` addresses.
UPGRADE_FLAGS = 0x11F64                    # 0x0002 boulder pocket open, 0x0200 statue, 0x0400 stump, 0x0800/0x1000 hoe
FAMILY_66 = 0x11F66                        # 0x2000 fish returned, 0x4000 hawker's hen trade
HOUSE_6A = 0x11F6A                         # 0x0040 cave berry eaten
FESTIVAL_6C = 0x11F6C                      # 0x1000 turtle, 0x2000 house berry armed, 0x4000 eaten, 0x8000 harvest festival
WEATHER = 0x0196
UNCLEAR_WEATHER = 0x001A                   # rain 0x02, snow 0x08, 0x10: the statue and the fishing camp refuse these days
RAIN_OR_SNOW = 0x000A                      # the animal shop's counter refuses these

STATUE_BIT, STUMP_BIT = 0x0200, 0x0400
BOULDER_OPEN_BIT = 0x0002
FISH_BIT, HAWKER_BIT = 0x2000, 0x4000
CAVE_BIT = 0x0040
EGG_TURTLE_BIT, EGG_HOUSE_ARM_BIT = 0x1000, 0x2000
EGG_PRIZE_BITS = EGG_TURTLE_BIT | EGG_HOUSE_ARM_BIT
HARVEST_FESTIVAL_BIT = 0x8000
SNOW_SEED_BOUGHT_ADDR, SNOW_SEED_BOUGHT_BIT = 0x11F6E, 0x0800
SNOW_SEED_TOOL, SNOW_SEED_SHELF_BIT = 0x17, 0x40

HARVEST_FESTIVAL_DAY, EGG_FESTIVAL_DAY = 12, 20
# Y1 Spring D2 and D3 are the farm clear and the first seed trips; the statue and the fishing camp wait for D4.
FIRST_FREE_SPRING_DAY = 4


@dataclass(frozen=True)
class BerrySource:
    name: str
    timeout: int        # ``PowerBerryTask`` steps the phase may run
    latest_hour: int    # latest clock hour the trip may still start


SOURCES: dict[str, BerrySource] = {s.name: s for s in (
    BerrySource("egg_festival", 45_000, 9),
    BerrySource("harvest_festival", 40_000, 9),
    BerrySource("cave", 40_000, 10),
    BerrySource("statue", 15_000, 10),
    BerrySource("fish", 30_000, 9),
    BerrySource("stump", 45_000, 9),
    BerrySource("chicken_trade", 30_000, 9),
    BerrySource("snow_flower", 30_000, 9),
)}


def enabled() -> bool:
    """True when this run collects scheduled power berries (``score`` and up, or ``HARVEST_POWER_BERRIES=1``)."""
    forced = os.environ.get(ENV_VAR)
    if forced is not None and forced.strip() != "":
        return forced.strip() not in {"0", "false", "no"}
    from harvest.planner import run_strategy

    return bool(run_strategy.active().collect_power_berries)


HARVEST_FESTIVAL_ENV = "HARVEST_HARVEST_FESTIVAL_BERRY"


def harvest_festival_enabled() -> bool:
    """The Harvest Festival fish trip; ``=0`` keeps the pre-exit-fix calendar for an A/B."""
    return os.environ.get(HARVEST_FESTIVAL_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _u16(ram, address: int) -> int:
    from harvest.core.ram_catalog import read_ram_u16

    return int(read_ram_u16(ram, address))


def _value(ram, name: str) -> int:
    from harvest.core.ram_catalog import read_ram_value

    return int(read_ram_value(ram, name))


def berry_receipts(ram) -> dict:
    """Read eaten totals and source bits together: a spawn bit alone is not an eat."""
    flags = {name: _u16(ram, address) for name, address in (
        ("upgrade", UPGRADE_FLAGS), ("family", FAMILY_66),
        ("house", HOUSE_6A), ("festival", FESTIVAL_6C),
    )}
    spent = ((flags["upgrade"] & 0x1E00).bit_count()
             + (flags["family"] & 0x6000).bit_count()
             + (flags["house"] & CAVE_BIT).bit_count()
             + (flags["festival"] & 0xC000).bit_count())
    return {"count": _value(ram, "power_berry_count"),
            "max_stamina": _value(ram, "max_stamina"),
            "spent_sources": spent, "flags": flags,
            "snow_seed_bought": bool(_u16(ram, SNOW_SEED_BOUGHT_ADDR) & SNOW_SEED_BOUGHT_BIT),
            "snow_seed_owned": snow_seed_owned(ram)}


def adult_hens(ram) -> int:
    from harvest.core.animal_status import count_chicken_slots

    return int(count_chicken_slots(ram)[0])


def snow_seed_owned(ram) -> bool:
    from harvest.core.carry import carry_pair_items
    from harvest.core.ram_catalog import read_ram_u8
    return SNOW_SEED_TOOL in carry_pair_items(ram) or bool(read_ram_u8(ram, SHED_ROW_2) & SNOW_SEED_SHELF_BIT)


def snow_seed_due(ram) -> bool:
    """The one-time festival offer, even after Eve no longer needs the dance."""
    return (enabled() and (_value(ram, "season"), _value(ram, "day")) == (0, 23)
            and _value(ram, "hour") < 15 and _value(ram, "money") >= 2000
            and _value(ram, "power_berry_count") < BERRY_TARGET
            and not _u16(ram, SNOW_SEED_BOUGHT_ADDR) & SNOW_SEED_BOUGHT_BIT
            and not snow_seed_owned(ram))


SHED_ROW_2 = 0x11F02                       # bit 0x04: the shelf holds the gold hammer, which replaces the plain one
GOLD_HAMMER_SHELF_BIT = 0x04
GOLD_STATUE_ENV = "HARVEST_GOLD_STATUE"


def statue_hammer_available(ram) -> bool:
    """False once the plain hammer is gone for the gold one: that trip fetches nothing and burns the day's berry slot.

    ROM trial (Year 3 pin, 2026-11-05): with the gold hammer the berry spawns (permanent bit 0x0200 flips) but the pickup
    recipe never catches it, so nothing is eaten and the source is spent; with the plain hammer the trip fails at the shelf
    (``tool 0x03 missing``) every day. ``HARVEST_GOLD_STATUE=1`` opts back in while the pickup is repaired.
    """
    from harvest.core.ram_catalog import read_ram_u8
    from harvest.core.carry import carry_pair_items
    forced = os.environ.get(GOLD_STATUE_ENV, "").strip().lower() in {"1", "true", "yes"}
    gold = bool(read_ram_u8(ram, SHED_ROW_2) & GOLD_HAMMER_SHELF_BIT) or 0x13 in carry_pair_items(ram)
    return forced or not gold


def due_source(ram, *, shops_closed: bool = False, force: bool = False) -> Optional[str]:
    """The one berry source worth a trip this morning, or ``None``.

    ``shops_closed`` is the plan's Sunday / festival / shut-day flag (the hen trade goes through the animal shop).
    ``force`` skips the strategy switch (tests). Reads RAM only: no hidden state, so a re-plan mid-day agrees.
    """
    if not force and not enabled():
        return None
    year, season, day, hour = (_value(ram, k) for k in ("year", "season", "day", "hour"))
    flags_64, flags_66 = _u16(ram, UPGRADE_FLAGS), _u16(ram, FAMILY_66)
    flags_6a, flags_6c = _u16(ram, HOUSE_6A), _u16(ram, FESTIVAL_6C)
    clear = not _u16(ram, WEATHER) & UNCLEAR_WEATHER

    def starts(name: str) -> bool:
        return hour <= SOURCES[name].latest_hour

    if season == SEASON_FALL and day == EGG_FESTIVAL_DAY:
        # The second prize still pays 21 points after all ten berries are eaten.
        return "egg_festival" if starts("egg_festival") and flags_6c & EGG_PRIZE_BITS != EGG_PRIZE_BITS else None
    if _value(ram, "power_berry_count") >= BERRY_TARGET:
        return None
    if season == SEASON_SPRING and day == 23:
        return None  # reserve the dated seed seller and dance, before 15:00
    if season == SEASON_WINTER and snow_seed_owned(ram) and starts("snow_flower"):
        return "snow_flower"
    if season == SEASON_FALL and day == HARVEST_FESTIVAL_DAY:
        # FestivalEscapeTask finishes the conversations, dance invitation and partner choice;
        # map 0x09 has no edge exit and its clock is frozen.
        if not harvest_festival_enabled():
            return None
        return "harvest_festival" if starts("harvest_festival") and not flags_6c & HARVEST_FESTIVAL_BIT else None
    # The pocket stays open for good once the first Winter snowstorm sets the boulder flag (ROM receipt: the recipe is
    # green from a Spring pin), so a cave berry missed in Winter is still worth a trip in any later season.
    if flags_64 & BOULDER_OPEN_BIT and not flags_6a & CAVE_BIT and starts("cave"):
        return "cave"
    first_free = year > 0 or season > SEASON_SPRING or day >= FIRST_FREE_SPRING_DAY
    if first_free and clear and not flags_64 & STATUE_BIT and starts("statue") and statue_hammer_available(ram):
        return "statue"
    if first_free and clear and season != SEASON_WINTER and not flags_66 & FISH_BIT and starts("fish"):
        return "fish"
    if (
        season == SEASON_FALL
        and not flags_66 & HAWKER_BIT
        and not shops_closed
        and not _u16(ram, WEATHER) & RAIN_OR_SNOW
        and adult_hens(ram) >= 2          # the route keeps two laying hens; the trade takes one
        and starts("chicken_trade")
    ):
        return "chicken_trade"
    # House wood can finish without ever rolling this berry. Once both field
    # berries, the statue and fish are done, spend a trip on the remaining stump.
    early = STATUE_BIT | 0x1800
    if (first_free and clear and flags_64 & early == early and flags_66 & FISH_BIT
            and not flags_64 & STUMP_BIT and starts("stump")):
        return "stump"
    return None


def power_berry_phases(source: str) -> list[PhaseSpec]:
    """The day-plan phase for ``source``: optional, so a failed trip costs the trip and never the day."""
    spec = SOURCES[source]
    return [PhaseSpec(
        f"COLLECT_POWER_BERRY_{source.upper()}", "power_berry",
        {"source": source, "timeout": spec.timeout, **({"ends_day": True} if source == "egg_festival" else {})},
        failure_policy="optional",
    )]


__all__ = [
    "BERRY_TARGET", "SOURCES", "BerrySource", "adult_hens", "due_source", "enabled", "power_berry_phases",
]
