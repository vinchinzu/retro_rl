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
    SEASON_FALL: ("eggplant",),
    SEASON_WINTER: (),
}

DEFAULT_SEASONAL_SEED: dict[int, str] = {
    SEASON_SPRING: "potato",
    SEASON_SUMMER: "corn",
    SEASON_FALL: "eggplant",
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

# Town festivals by (season_id, day_of_month):
# Spring: D8 (Flower Festival), D23 (Horse Race)
# Summer: D1 (Fireworks), D20 (Cow Festival)
# Fall: D12 (Harvest Festival), D20 (Egg Festival)
# Winter: D10 (Thanksgiving), D24 (Star Night), D30 (New Year's Eve)
FESTIVAL_CALENDAR: dict[tuple[int, int], str] = {
    (SEASON_SPRING, 8): "Flower Festival",
    (SEASON_SPRING, 23): "Horse Race",
    (SEASON_SUMMER, 1): "Fireworks",
    (SEASON_SUMMER, 20): "Cow Festival",
    (SEASON_FALL, 12): "Harvest Festival",
    (SEASON_FALL, 20): "Egg Festival",
    (SEASON_WINTER, 10): "Thanksgiving",
    (SEASON_WINTER, 24): "Star Night",
    (SEASON_WINTER, 30): "New Year's Eve",
}

# Task recording mappings for festivals
FESTIVAL_RECORDINGS: dict[tuple[int, int], str] = {
    (SEASON_SPRING, 8): "spring_festival",
    (SEASON_SPRING, 23): "spring_festival",
    (SEASON_SUMMER, 1): "spring_festival",
    (SEASON_SUMMER, 20): "spring_festival",
    (SEASON_FALL, 12): "spring_festival",
    (SEASON_FALL, 20): "spring_festival",
    (SEASON_WINTER, 10): "spring_festival",
    (SEASON_WINTER, 24): "spring_festival",
    (SEASON_WINTER, 30): "spring_festival",
}


def is_festival_day(season: int | str, day: int) -> bool:
    """Return True if (season, day) is a town festival day."""
    s = normalize_season(season)
    return (s, int(day)) in FESTIVAL_CALENDAR


def festival_name_for_date(season: int | str, day: int) -> Optional[str]:
    """Return festival name for (season, day) or None."""
    s = normalize_season(season)
    return FESTIVAL_CALENDAR.get((s, int(day)))


def festival_recording_for_date(season: int | str, day: int) -> str:
    """Return recorded task name for (season, day) festival."""
    s = normalize_season(season)
    return FESTIVAL_RECORDINGS.get((s, int(day)), "spring_festival")


ATTEND_FESTIVAL_PHASE = PhaseSpec(
    "ATTEND_FESTIVAL",
    PhaseKind.RECORDED,
    {"task_name": "spring_festival"},
    failure_policy="optional",
)


def attend_festival_phase(season: int | str, day: int) -> PhaseSpec:
    """Build a PhaseSpec to attend today's festival in town."""
    s = normalize_season(season)
    d = int(day)
    name = festival_name_for_date(s, d) or "Festival"
    task_name = festival_recording_for_date(s, d)
    return PhaseSpec(
        "ATTEND_FESTIVAL",
        PhaseKind.RECORDED,
        {
            "task_name": task_name,
            "festival_name": name,
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


# ── Storms: Hurricanes (Summer) & Blizzards (Winter) ────────────────────

WEATHER_SUNNY = 0
WEATHER_RAIN = 1
WEATHER_SNOW = 2
WEATHER_HURRICANE = 3
WEATHER_BLIZZARD = 3


def is_storm_weather(weather_code: int, season: int = 0) -> bool:
    """Return True if weather_code indicates a hurricane or blizzard.

    In HM SNES, weather code 3 triggers hurricane (summer) or blizzard (winter).
    During storms, the player is trapped indoors and cannot leave the farmhouse.
    """
    return int(weather_code) == WEATHER_HURRICANE


def is_storm_day(
    season: int | str,
    weather_code: Optional[int] = None,
    *,
    is_hurricane: Optional[bool] = None,
    is_blizzard: Optional[bool] = None,
) -> bool:
    """Return True if today is a storm day confining the farmer indoors."""
    if is_hurricane or is_blizzard:
        return True
    if weather_code is not None:
        s = normalize_season(season)
        return is_storm_weather(weather_code, s)
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
    "FESTIVAL_RECORDINGS",
    "is_festival_day",
    "festival_name_for_date",
    "festival_recording_for_date",
    "ATTEND_FESTIVAL_PHASE",
    "attend_festival_phase",
    "SUNDAY_CHURCH_PHASE",
    "SUNDAY_MOUNTAIN_PHASE",
    "WEATHER_SUNNY",
    "WEATHER_RAIN",
    "WEATHER_SNOW",
    "WEATHER_HURRICANE",
    "WEATHER_BLIZZARD",
    "is_storm_weather",
    "is_storm_day",
    "REPAIR_CROPS_PHASE",
    "post_storm_recovery_phases",
]
