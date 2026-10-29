"""Calendar, festival, Sunday, storm, and seasonal crop rotation helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseKind, PhaseSpec
from harvest.planner.crop_planner import (
    SEASON_SPRING,
    SEASON_SUMMER,
    SEASON_FALL,
    SEASON_WINTER,
    normalize_season,
)

# ── Seasonal Crop Rotations ──────────────────────────────────────────────

SEASONAL_CROPS: dict[int, tuple[str, ...]] = {
    SEASON_SPRING: ("potato", "turnip"),
    SEASON_SUMMER: ("corn", "tomato"),
    # The ROM has no fall seed (seed counters: grass/corn/tomato/potato/turnip).
    SEASON_FALL: (),
    SEASON_WINTER: (),
}

DEFAULT_SEASONAL_SEED: dict[int, str] = {
    SEASON_SPRING: "potato",
    SEASON_SUMMER: "corn",
}


def seasonal_crops_for_season(season: int | str) -> tuple[str, ...]:
    """Return valid crops for the given season index (0=spring..3=winter) or name."""
    s = normalize_season(season)
    return SEASONAL_CROPS.get(s, ())


def default_crop_for_season(
    season: int | str,
    *,
    preferred: Optional[str] = None,
) -> Optional[str]:
    """Select the crop for a season, keeping preferred crop if valid in season."""
    s = normalize_season(season)
    crops = SEASONAL_CROPS.get(s, ())
    if not crops:
        return None
    if preferred and preferred in crops:
        return preferred
    return DEFAULT_SEASONAL_SEED.get(s, crops[0])


def next_crop_in_rotation(
    season: int | str,
    current_crop: Optional[str] = None,
) -> Optional[str]:
    """Rotate to the next crop within the season, or the season default."""
    s = normalize_season(season)
    crops = SEASONAL_CROPS.get(s, ())
    if not crops:
        return None
    if current_crop in crops:
        idx = (crops.index(current_crop) + 1) % len(crops)
        return crops[idx]
    return crops[0]


# ── Festival Calendar ───────────────────────────────────────────────────

# ROM-verified (bank_82 WeatherTomorrow $828CF9, measured on pins 2026-09-28).
# Each night the ROM computes the NEXT day's forecast byte ($098C) from today's
# date: `day == N` writes the festival code for D(N+1). SetWeatherFlags copies
# it at rollover and, for codes >= 6, leaves today's flags ($0196) empty, so a
# festival morning looks like plain calm weather. Proof: Y1_D22_Town_Eve holds
# forecast 6 at D22 18:13 and the recorded festival ends on D24 morning.
#
# There is NO Summer festival, no Spring D8 festival, no Horse Race, no
# Fireworks, no Cow Festival and no "New Year's Eve" on Winter D30 in this
# ROM. Winter D30's eve forecasts the New Year (code 0x0A) for the next Spring
# D1, and only when that rollover increments the year (Y2+).
FORECAST_FESTIVAL_CODES: dict[tuple[int, int], int] = {
    (SEASON_SPRING, 23): 0x06,  # Flower Festival
    (SEASON_FALL, 12): 0x07,  # Harvest Festival
    (SEASON_WINTER, 10): 0x08,  # Thanksgiving
    (SEASON_WINTER, 24): 0x09,  # Star Night
    (SEASON_SPRING, 1): 0x0A,  # New Year (Y2+ only)
    (SEASON_FALL, 20): 0x0B,  # Egg Festival
}

FESTIVAL_CALENDAR: dict[tuple[int, int], str] = {
    (SEASON_SPRING, 23): "Flower Festival",
    (SEASON_FALL, 12): "Harvest Festival",
    (SEASON_FALL, 20): "Egg Festival",
    (SEASON_WINTER, 10): "Thanksgiving",
    (SEASON_WINTER, 24): "Star Night",
}
NEW_YEAR_DATE = (SEASON_SPRING, 1)
NEW_YEAR_NAME = "New Year"


def festival_name_for_date(
    season: int | str, day: int, year: int = 0
) -> Optional[str]:
    """Festival name for a ROM ``year`` (0-based, as in RAM) date, or None."""
    s = normalize_season(season)
    if (s, int(day)) == NEW_YEAR_DATE:
        return NEW_YEAR_NAME if int(year) >= 1 else None
    return FESTIVAL_CALENDAR.get((s, int(day)))


def is_festival_day(season: int | str, day: int, year: int = 0) -> bool:
    """Return True if the date is a town festival day (see table note)."""
    return festival_name_for_date(season, day, year) is not None


def forecast_code_for_date(year: int, season: int | str, day: int) -> int:
    """ROM forecast byte the night before this date, for festival dates else 0."""
    if festival_name_for_date(season, day, year) is None:
        return 0
    return FORECAST_FESTIVAL_CODES[(normalize_season(season), int(day))]


ATTEND_FESTIVAL_PHASE = PhaseSpec(
    "ATTEND_FESTIVAL",
    PhaseKind.RECORDED,
    {"task_name": "spring_festival"},
    failure_policy="optional",
)


def attend_festival_phase(season: int | str, day: int, year: int = 0) -> PhaseSpec:
    """Build a PhaseSpec to attend today's festival in town."""
    s = normalize_season(season)
    d = int(day)
    if (s, d) == (SEASON_SPRING, 23):  # talk to everyone, dance with Eve; the finale ends the day
        return PhaseSpec(
            "ATTEND_FESTIVAL",
            PhaseKind.FLOWER_FESTIVAL,
            {"festival_name": "Flower Festival", "season": s, "day": d, "ends_day": True},
            failure_policy="optional",
        )
    return PhaseSpec(
        "ATTEND_FESTIVAL",
        PhaseKind.RECORDED,
        {
            "task_name": "spring_festival",
            "festival_name": festival_name_for_date(s, d, year) or "Festival",
            "season": s,
            "day": d,
        },
        failure_policy="optional",
    )


# ── Sunday Phases ───────────────────────────────────────────────────────

SUNDAY_CHURCH_PHASE = PhaseSpec(
    "SUNDAY_CHURCH",
    PhaseKind.RECORDED,
    {"task_name": "sunday_go_to_church"},
    failure_policy="optional",
)

SUNDAY_MOUNTAIN_PHASE = PhaseSpec(
    "SUNDAY_MOUNTAIN",
    PhaseKind.RECORDED,
    {"task_name": "sunday_go_to_mountain"},
    failure_policy="optional",
)


# ── Storms: hurricanes (Summer only) ────────────────────────────────────

# $098C forecast codes. Code 3 comes only from Hurricane_Chance_Table, which is
# non-zero for Summer alone: the ROM has no blizzard. Winter snow is code 2 (or
# rain) and lands in $0196 as 0x08; Year-1 Winter can also roll the one-off
# "fair" (code 4, $0196 0x200).
WEATHER_SUNNY = 0
WEATHER_RAIN = 1
WEATHER_SNOW = 2
WEATHER_HURRICANE = 3
WEATHER_FAIR = 4


def is_storm_weather(weather_code: int, season: int = 0) -> bool:
    """True when a FORECAST byte says hurricane (the eve, not the day itself)."""
    return int(weather_code) == WEATHER_HURRICANE


def is_storm_day(
    season: int | str,
    weather_code: Optional[int] = None,
    *,
    is_hurricane: Optional[bool] = None,
) -> bool:
    """Return True if today is a hurricane day confining the farmer indoors."""
    if is_hurricane:
        return True
    if weather_code is not None:
        return is_storm_weather(weather_code, normalize_season(season))
    return False


REPAIR_CROPS_PHASE = PhaseSpec(
    "REPAIR_CROPS",
    PhaseKind.RECORDED,
    {"task_name": "repair_crops"},
    failure_policy="optional",
)


def post_storm_recovery_phases(season: int | str = 0, day: int = 1) -> List[PhaseSpec]:
    """Phases executed the morning after a storm: debris clearing and fence repair."""
    from harvest.planner.day_phase_catalog import CLEAR_FIELD_PHASE

    return [
        CLEAR_FIELD_PHASE,
        REPAIR_CROPS_PHASE,
    ]


__all__ = [
    "SEASONAL_CROPS",
    "DEFAULT_SEASONAL_SEED",
    "seasonal_crops_for_season",
    "default_crop_for_season",
    "next_crop_in_rotation",
    "FESTIVAL_CALENDAR",
    "FORECAST_FESTIVAL_CODES",
    "NEW_YEAR_DATE",
    "forecast_code_for_date",
    "is_festival_day",
    "festival_name_for_date",
    "ATTEND_FESTIVAL_PHASE",
    "attend_festival_phase",
    "SUNDAY_CHURCH_PHASE",
    "SUNDAY_MOUNTAIN_PHASE",
    "WEATHER_SUNNY",
    "WEATHER_RAIN",
    "WEATHER_SNOW",
    "WEATHER_HURRICANE",
    "WEATHER_FAIR",
    "is_storm_weather",
    "is_storm_day",
    "REPAIR_CROPS_PHASE",
    "post_storm_recovery_phases",
]
