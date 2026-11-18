"""Relationship calendar: which girls to court today, how far, and where.

The ranch score pays ``(hearts & 0x1FF) >> 4`` for each of the five girls: 31 points at 496-511 and next to nothing
past 511, where the nine-bit mask wraps. The wife keeps her own routine (``tasks/eve_gift``); this module grows the
other four toward the window and stops short of it by whatever the calendar will still add on its own.

ROM facts it rests on (event scripts read with ``harvest.scripts.event_dis``; receipts in ``tasks/relationships``):

- **Sunday church** (group 3 event 1, ``CODE_83D689``): on a clear Sunday after the first church visit
  (``$7F1F68`` bit 0) Maria and Ann each ask one question per church entry, no season, heart or hour test:
  choice 0 is +4. Measured on ``PR_f1ctl_s1_Y1_Fall_D3``: 20 entries, Maria and Ann 8 -> 88 each, 42,499 frames,
  the clock 07:11 -> 07:13 (a re-entry is a few outdoor frames). Rain, snow, the hurricane eve and ``$0196 & 0x200``
  load other church scripts that pay nothing.
- **The 200-heart event** (``CODE_83CC72``, 06:00 on days 2-7, 13-17 and 25-28): fires only while the farmer is
  unmarried, for the first unblocked girl at 200+ (Maria, Ann, Nina, Ellen, Eve order; Maria then Ann both fired in
  ``f4all``), and its flag
  (``$7F1F6A`` bit 12 Maria, bit 14 Ann) removes Maria from the Sunday church and silences Ann's question. Measured, the
  flags then step on (0x2000, 0x8000) and the questions pay again, but Ann lost a Sunday (``f4all``). So no girl is
  courted past ``PRE_WEDDING_CAP`` before the wedding.
- **Uncontrolled gains still ahead** count against the cap: the Flower Festival (Spring D23, the bot talks to every
  girl: +8 / +12 / +15 by tier), New Year (Years 2-3 Spring D1, +10 each girl talked to), the Egg Festival roam
  (Fall D20, +10 measured for Ellen) and Star Night (Winter D24: the
  default square scene pays Ellen +30; with Eve the wife the planner picks her scene and no other girl gains). The Harvest
  Festival's +30 scene (group 0x26) is not reached: the bot no longer enters that map.
"""
from __future__ import annotations

import math
import os
from typing import Optional

from harvest.planner.day_phase_types import PhaseSpec

ENV_VAR = "HARVEST_SOCIAL_CHURCH"         # 1/0 forces the church calendar on or off, whatever the strategy
SOCIAL_TARGET = 500                       # the window is 496-511; 500 leaves 11 of slack for a stray +8
PRE_WEDDING_CAP = 199                     # the 200-heart event fires only while unmarried
CHURCH_GAIN = 4                           # choice 0 of the Sunday question, Maria and Ann
CHURCH_GIRLS = ("maria", "ann")
CHURCH_MAX_ENTRIES = 30                   # ~2,100 frames an entry; the clock moves ~6 ticks
CHURCH_LATEST_START_HOUR = 15             # the door refuses from 17:00
CHURCH_MIN_ENTRIES = 3                    # a 1-entry visit is a ~4,700-frame walk for +4 (f6final); wait for more need
FLOWER_FESTIVAL_GAIN = 15                 # the 301+ tier; the lower tiers pay less, so this is the safe margin
STAR_NIGHT_GAIN = 30
STAR_NIGHT_GIRL = "ellen"                 # F11: the bot leaves the Star Night cursor on index 0
NEW_YEAR_GAIN = 10                        # group 0x28 events 1/3: +10 each girl; the New Year escape talks to everyone
EGG_FESTIVAL_GAIN = 10                    # the Egg Festival roam talked Ellen +10 (f2on, Y1 Fall D20); any girl can get it
HARVEST_QUESTION_GAIN = 8                 # G26.E0 first questions, before the dance (+30 to the partner)
MARRIAGE_BITS = {"maria": 0x01, "ann": 0x02, "nina": 0x04, "ellen": 0x08, "eve": 0x10}
WEDDED_MASK = 0x1F

WEATHER = 0x0196
CHURCH_WEATHER_BLOCK = 0x020A             # rain 0x02, snow 0x08 and 0x200 load other church scripts
STORM_FORECAST = 3                        # weather_tomorrow 3 (hurricane eve) loads the storm script
FESTIVAL_FORECAST = 6                     # 6+ (festival eves) load the weekday church script
UPGRADE_64, NO_ACTORS_64 = 0x11F64, 0x0020  # $7F1F64 bit 0x20 loads no church actors at all
ROMANCE_6A = 0x11F6A                      # 0x1000 Maria's 200-heart event, 0x4000 Ann's
EVENT_68 = 0x11F68                        # 0x0001 first church visit done
CHURCH_SILENCED = {"maria": 0x1000, "ann": 0x4000}

DAYS_PER_SEASON, SEASONS = 30, 4
END_INDEX = 2 * 120 + 1 * 30 + 29         # Year 3 Summer D30, the ending morning
FLOWER_FESTIVAL = (0, 23)
STAR_NIGHT = (3, 24)
NEW_YEAR = (0, 1)                         # Years 2-3 only; Year 1 Spring D1 is day 0, never after today
EGG_FESTIVAL = (2, 20)
HARVEST_FESTIVAL = (2, 12)


def enabled() -> bool:
    """True when the run courts the other girls (``score`` and up, or ``HARVEST_SOCIAL_CHURCH=1``)."""
    forced = os.environ.get(ENV_VAR)
    if forced is not None and forced.strip() != "":
        return forced.strip() not in {"0", "false", "no", "off"}
    from harvest.planner import run_strategy

    return bool(run_strategy.active().court_girls)


def _value(ram, name: str) -> int:
    from harvest.core.ram_catalog import read_ram_value

    return int(read_ram_value(ram, name))


def _u16(ram, address: int) -> int:
    from harvest.core.ram_catalog import read_ram_u16

    return int(read_ram_u16(ram, address))


def date_index(year: int, season: int, day: int) -> int:
    """0-based day count from Year 1 Spring D1 (``year`` is the 0-based RAM year)."""
    return int(year) * DAYS_PER_SEASON * SEASONS + int(season) * DAYS_PER_SEASON + int(day) - 1


def _events_after(today: int, season: int, day: int) -> int:
    """How many dates (season, day) fall strictly after ``today`` and before the ending."""
    count = 0
    for year in range(3):
        at = date_index(year, season, day)
        if today < at < END_INDEX:
            count += 1
    return count


def upcoming_gain(girl: str, year: int, season: int, day: int, *, wife: Optional[str] = None) -> int:
    """Hearts the calendar still adds to ``girl`` without a courting stop (the margin her cap keeps free)."""
    if girl == wife:
        return 0
    today = date_index(year, season, day)
    gain = (FLOWER_FESTIVAL_GAIN * _events_after(today, *FLOWER_FESTIVAL) + NEW_YEAR_GAIN * _events_after(today, *NEW_YEAR)
            + EGG_FESTIVAL_GAIN * _events_after(today, *EGG_FESTIVAL))
    from harvest.planner.power_berry_calendar import harvest_festival_enabled

    if harvest_festival_enabled() and _events_after(today, *HARVEST_FESTIVAL):
        # The berry receipt retires this trip after its first success. Reserve
        # at most one visit's questions; Year 3 tops up unused festival margins.
        gain += HARVEST_QUESTION_GAIN
    # The planner picks the wife's Star Night scene when it has stands for it (Eve: festival_escape); otherwise
    # the default square scene pays Ellen.
    from harvest.planner.tasks.festival_escape import STAR_NIGHT_SCENE_MAPS

    if girl == STAR_NIGHT_GIRL and wife not in STAR_NIGHT_SCENE_MAPS:
        gain += STAR_NIGHT_GAIN * _events_after(today, *STAR_NIGHT)
    return gain


def wife_of(ram) -> Optional[str]:
    married = _value(ram, "marriage_flags") & WEDDED_MASK
    return next((girl for girl, bit in MARRIAGE_BITS.items() if married & bit), None)


def heart_cap(ram, girl: str) -> int:
    """Hearts at which courting ``girl`` stops today."""
    wife = wife_of(ram)
    if wife is None:
        return PRE_WEDDING_CAP
    year, season, day = (_value(ram, key) for key in ("year", "season", "day"))
    return SOCIAL_TARGET - upcoming_gain(girl, year, season, day, wife=wife)


def church_open_for_girls(ram) -> bool:
    """The clear-Sunday church script loads (``CODE_83D689``) and the door still has time before 17:00."""
    if _value(ram, "weekday") != 0 or _value(ram, "hour") > CHURCH_LATEST_START_HOUR:
        return False
    forecast = _value(ram, "weather_tomorrow")
    if _u16(ram, WEATHER) & CHURCH_WEATHER_BLOCK or forecast == STORM_FORECAST or forecast >= FESTIVAL_FORECAST:
        return False
    if _u16(ram, UPGRADE_64) & NO_ACTORS_64:
        return False
    from harvest.calendar_schedule import festival_on

    if festival_on(_value(ram, "year"), _value(ram, "season"), _value(ram, "day")) is not None:
        return False
    return bool(_u16(ram, EVENT_68) & 0x0001)


def church_targets(ram) -> dict[str, int]:
    """Girl -> cap for every church girl still a question short of her cap."""
    wife = wife_of(ram)
    silenced = _u16(ram, ROMANCE_6A)
    targets = {}
    for girl in CHURCH_GIRLS:
        if girl == wife or silenced & CHURCH_SILENCED[girl]:
            continue
        cap = heart_cap(ram, girl)
        if _value(ram, f"{girl}_hearts") + CHURCH_GAIN <= cap:
            targets[girl] = cap
    return targets


def church_phase(ram, *, force: bool = False) -> Optional[PhaseSpec]:
    """Today's Sunday church visit, or ``None``: one entry per +4 the neediest girl still lacks."""
    if not force and not enabled():
        return None
    if not church_open_for_girls(ram):
        return None
    targets = church_targets(ram)
    if not targets:
        return None
    need = max(cap - _value(ram, f"{girl}_hearts") for girl, cap in targets.items())
    entries = min(CHURCH_MAX_ENTRIES, math.ceil(need / CHURCH_GAIN))
    if entries < CHURCH_MIN_ENTRIES and _value(ram, "year") < 2:
        return None   # top-ups wait for the final year, when no festival is left to fill the gap
    return PhaseSpec(
        "SOCIAL_CHURCH",
        "church_court",
        {"targets": targets, "max_entries": entries, "timeout": 2_400 * entries + 30_000},
        failure_policy="optional",
    )


# Weekday town questions (tasks/relationships.TownCourtTask): Ellen in Spring and Fall, Nina in Summer and Winter.
TOWN_COURT_SEASONS = {"ellen": (0, 2), "nina": (1, 3)}
TOWN_COURT_GAIN = 8
WINDOW_FLOOR = 496                         # 31 heart points start here (hearts >> 4); 511 is the last before the nine-bit wrap
# Crop seasons keep the morning for the crops: two loops, and only for a girl far behind her cap.
TOWN_COURT_MAX_LOOPS = {0: 2, 1: 2, 2: 8, 3: 8}
TOWN_COURT_BEHIND = 80
TOWN_COURT_LATEST_START_HOUR = 13
TOWN_WEATHER_BLOCK = 0x020A                # CODE_83D46D loads no townsfolk in rain, snow or 0x200
WINTER_SHOPS_SHUT_DAY = 10                 # door flag 0x10 shuts the shops on Winter D10


def town_open_for_girls(ram) -> bool:
    """The weekday town script loads its girls and the shops they stand beside are open."""
    weekday, season, day = (_value(ram, key) for key in ("weekday", "season", "day"))
    # The opening farm still has its fences and boulders. A town detour from
    # the house strands the forage/shop route below them before D2 clears it.
    if _value(ram, "year") == 0 and season == 0 and day <= 2:
        return False
    if weekday not in range(1, 6) or _value(ram, "hour") > TOWN_COURT_LATEST_START_HOUR:
        return False
    if season == 3 and day == WINTER_SHOPS_SHUT_DAY:
        return False
    forecast = _value(ram, "weather_tomorrow")
    # A festival eve (forecast 6+) gives the girls festival lines: 8 talks paid nothing on Y1 Fall D11 (f3all).
    if _u16(ram, WEATHER) & TOWN_WEATHER_BLOCK or forecast == STORM_FORECAST or forecast >= FESTIVAL_FORECAST:
        return False
    if _u16(ram, UPGRADE_64) & NO_ACTORS_64:
        return False
    from harvest.calendar_schedule import festival_on

    return festival_on(_value(ram, "year"), season, day) is None


def town_court_phase(ram, *, force: bool = False) -> Optional[PhaseSpec]:
    """Today's weekday town visit for the girl whose question this season asks, or ``None``."""
    if not force and not enabled():
        return None
    if not town_open_for_girls(ram):
        return None
    season = _value(ram, "season")
    wife = wife_of(ram)
    for girl, seasons in TOWN_COURT_SEASONS.items():
        if season not in seasons or girl == wife:
            continue
        cap = heart_cap(ram, girl)
        need = cap - _value(ram, f"{girl}_hearts")
        # Year 3 Spring/Summer is the girl's last window (no Fall or Winter follows), so any shortfall is worth a visit.
        final_year = _value(ram, "year") >= 2
        hearts = _value(ram, f"{girl}_hearts")
        if final_year and hearts < WINDOW_FLOOR <= cap and need < TOWN_COURT_GAIN:
            # y3m: Nina ended at 495 of the 496 window floor and cap 500 left 5 hearts of "need", under one +8 visit.
            # No wrap is near (495 + 8 = 503), so the last window takes the visit rather than the lost point.
            need = TOWN_COURT_GAIN
            cap = hearts + TOWN_COURT_GAIN
        if need < TOWN_COURT_GAIN or (season in (0, 1) and need < TOWN_COURT_BEHIND and not final_year):
            continue
        loops = min(TOWN_COURT_MAX_LOOPS[season], math.ceil(need / TOWN_COURT_GAIN))
        return PhaseSpec(
            "SOCIAL_TOWN",
            "town_court",
            {"girl": girl, "cap": cap, "loops": loops, "timeout": 4_000 * loops + 30_000},
            failure_policy="optional",
        )
    return None


def ring_corn_trip_first(ram) -> bool:
    """Summer D1-D3 on a shop-open weekday with ripe crops due: the rings' corn bags are the day's one town trip.

    A ring planted by D3 gets six harvests (``RING_FIRST_PLANT_DAYS``). The harvest frees the rings, the bags are
    bought after it, and the shop shuts at 17:00 with the weekend next. Courting first took the trip (Y1 Summer D3
    of ``compose5``: Nina until 16:01) and the rings stayed bare, a whole Summer of crop income lost. Only the
    score family caps shipping, so the cash strategy keeps its old order.
    """
    from harvest.planner import run_strategy
    from harvest.planner.day_plan_status import SHOP_CLOSED_WEEKDAYS
    from harvest.planner.livestock_econ import RING_FIRST_PLANT_DAYS

    return (_value(ram, "season") == 1 and _value(ram, "day") <= RING_FIRST_PLANT_DAYS
            and _value(ram, "weekday") not in SHOP_CLOSED_WEEKDAYS
            and run_strategy.quota_harvest_due(ram))


SPRING_SOW_FIRST_FROM_YEAR = 2     # 0-based: Year 3
SPRING_SOW_FIRST_ENV = "HARVEST_Y3_COURT_YIELD"     # "0" restores the court-first order for an A/B
SPRING_SOW_FIRST_LAST_DAY = 3      # the opening days, when every site is bare


def spring_sowing_trip_first(ram) -> bool:
    """Year 3 Spring D1-D3 on a shop-open weekday with bare sites: the seed trip is the day's one town trip.

    ``fixD`` Y3 Spring D2 courted Ellen first (4,189 frames), reached the seed window's 13:11 cutoff and bought no
    potato or turnip seed, so twelve bare sites waited a day for the D3 trip. Year 3 Spring is Ellen's last
    window, so the visit is not lost: any shortfall is courted on the first later weekday with room for it.
    """
    from harvest.maps.farm_pond import active_plot_centers, plot_crop_tiles
    from harvest.planner import run_strategy
    from harvest.planner.day_plan_status import SHOP_CLOSED_WEEKDAYS
    from harvest.planner.field_cover import farm_tiles
    from harvest.tasks.crop_skills import PLANTED_OR_WET

    if not (os.environ.get(SPRING_SOW_FIRST_ENV, "1") != "0"
            and _value(ram, "year") >= SPRING_SOW_FIRST_FROM_YEAR and _value(ram, "season") == 0
            and _value(ram, "day") <= SPRING_SOW_FIRST_LAST_DAY
            and _value(ram, "weekday") not in SHOP_CLOSED_WEEKDAYS
            and run_strategy.active().caps_shipping
            and run_strategy.one_shot_sowing_ok(ram)
            and not _value(ram, "potato_seeds") and not _value(ram, "turnip_seeds")):
        return False
    # The morning plan is made in the house, where the live map is not the farm: read the persistent farm array.
    tiles = farm_tiles(ram)
    year = _value(ram, "year")
    return any(
        sum(int(tiles[y * 64 + x]) in PLANTED_OR_WET for x, y in plot_crop_tiles(center, year)) == 0
        for center in active_plot_centers(year)
    )


COURT_FIRST_ENV = "HARVEST_Y3_COURT_FIRST"     # "0" restores harvest-before-court for an A/B


def court_before_harvest(ram, phase: Optional[PhaseSpec]) -> bool:
    """Year 3 Spring weekday court: walk to town before the ripe-crop harvest, not after it.

    Ellen's last window is Y3 Spring. With twenty sites some crop is ripe nearly every morning, the harvest runs to the
    17:00 bin deadline and the court then starts at 17:xx among closed doors (``y3k_b``: loops=0 talks=0 on D27, D28 and
    D29, Ellen stuck at 484 for 1 heart point). The crops keep on the plant, so the visit goes first.
    """
    return (phase is not None and phase.phase == "SOCIAL_TOWN" and os.environ.get(COURT_FIRST_ENV, "1") != "0"
            and _value(ram, "year") >= SPRING_SOW_FIRST_FROM_YEAR and _value(ram, "season") == 0)


def social_phase(ram) -> Optional[PhaseSpec]:
    """The day's courting stop: the Sunday church, else a weekday town question."""
    if ring_corn_trip_first(ram) or spring_sowing_trip_first(ram):
        return None
    return church_phase(ram) or town_court_phase(ram)


__all__ = [
    "CHURCH_GAIN",
    "CHURCH_MAX_ENTRIES",
    "PRE_WEDDING_CAP",
    "SOCIAL_TARGET",
    "church_open_for_girls",
    "church_phase",
    "church_targets",
    "date_index",
    "enabled",
    "heart_cap",
    "court_before_harvest",
    "ring_corn_trip_first",
    "social_phase",
    "spring_sowing_trip_first",
    "town_court_phase",
    "town_open_for_girls",
    "upcoming_gain",
    "wife_of",
]
