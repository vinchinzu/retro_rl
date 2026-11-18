"""Mountain forage by season: what the ROM spawns, what it pays, and whether a trip fits the day.

ROM facts (HM-Decomp ``DATA8_82CFB4`` in bank_82, ``Items_Price_Table`` in bank_81). A pickup tile
hands over held item ``table[tile * 4 + season]``. Mountain map 0x10 holds the same tiles in every
season, so what a tile gives depends on the calendar alone:

| tile | at (x, y)                          | Spring     | Summer           | Fall                | Winter |
| ---- | ---------------------------------- | ---------- | ---------------- | ------------------- | ------ |
| 0x30 | (2,4) (45,5) (23,9) (20,26)        | grapes 150 | nothing          | nothing             | -      |
| 0x32 | (28,10)                            | nothing    | green fruit 200  | nothing             | -      |
| 0x33 | (1,4) (46,13) (30,24) (14,36) (41,40) | nothing | nothing          | mushroom 150        | -      |
| 0x34 | (43,39)                            | nothing    | nothing          | poison mushroom 200 | -      |

A pickup tile gives again the same day (two grape trips and two green-fruit trips shipped twice).

(Tile 0x31 gives a flower in Spring and Summer and tile 0x36 an herb, 200G, in every season, but
neither stands on the maps the bot walks.) Measured 2026-11-02 on the Summer D5 pin
``LN_cowfix_poweron_summer_d5``: the grape stand and the mushroom stand both come back empty
("unverified after 3 picks", held 0), which is why the grape run was gated to Spring.

The farmer carries one item and the bin pays only before 17:00, so a trip is one house-to-bin loop.
The clock runs 900 frames an hour and then freezes after 18:xx while work continues, so a morning
loop costs frames, not crop work, on a day with nothing to ship before 17:00.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

SEASON_SPRING, SEASON_SUMMER, SEASON_FALL, SEASON_WINTER = 0, 1, 2, 3

TICKS_PER_HOUR = 900
BIN_CLOSE_HOUR = 17  # ShippingScene: a forage item put in the bin later pays nothing
# Frames of slack left at the bin after a loop (route retries, the throw, the 17:00 check).
BIN_MARGIN_FRAMES = 600
# Latest hour a trip may still go ahead of the crop work. ``trips_today`` already refuses a loop that cannot reach the
# bin by 17:00; from Year 2 the wife gift, the wood trip and the herd chores start the outdoor plan at 09:00-12:00,
# so the old limit of 9 left two grape trips in Year 2 Spring (``y2c``: 34 berry windows hit the 14:00 cutoff).
FORAGE_FIRST_LATEST_HOUR = 12
# ``HARVEST_FORAGE_FIRST=0`` restores the old order: forage last, after all crop work.
FORAGE_FIRST = os.environ.get("HARVEST_FORAGE_FIRST", "1") != "0"


@dataclass(frozen=True)
class ForageKind:
    name: str  # task ``forage=`` key: "grape" | "green_fruit" | "mushroom"
    season: int
    item_id: int  # ROM held-item id (HM-Decomp TODO.txt item list)
    price_g: int  # ``Items_Price_Table`` ship price
    tiles: tuple[tuple[int, int], ...]  # mountain tiles that give it this season
    loop_frames: int  # measured house -> pick -> bin for one item
    max_per_day: int  # trips a day that the pick is known to repay


GRAPE = ForageKind("grape", SEASON_SPRING, 3, 150, ((20, 26), (23, 9), (2, 4), (45, 5)), 3_300, 2)
# The tile gives again the same day: two loops shipped 200G twice on the Summer D5 pin (home 10:30, 15:00).
GREEN_FRUIT = ForageKind("green_fruit", SEASON_SUMMER, 4, 200, ((28, 10),), 3_600, 2)
MUSHROOM = ForageKind(
    "mushroom", SEASON_FALL, 1, 150, ((41, 40), (30, 24), (14, 36), (46, 13), (1, 4)), 3_000, 1
)

# The poison mushroom (tile 0x34, Fall only) ships for 200G, 50G over the mushroom, and stands two tiles
# from the mushroom stand's tile, so it is Fall's trip; the mushroom stays as the fallback kind.
# Measured on ``PR_y1a_Y1_Fall_D10``: two trips shipped 200G twice in 4,545 frames (home 12:00), so a loop is
# ~2,250 frames, far shorter than the grape (the stand is by the south landing) and Fall has no crop work.
POISON_MUSHROOM = ForageKind("poison_mushroom", SEASON_FALL, 2, 200, ((43, 39),), 2_300, 4)

FORAGE_BY_SEASON: dict[int, ForageKind] = {
    SEASON_SPRING: GRAPE,
    SEASON_SUMMER: GREEN_FRUIT,
    SEASON_FALL: POISON_MUSHROOM,
}
ALL_KINDS: tuple[ForageKind, ...] = (GRAPE, GREEN_FRUIT, MUSHROOM, POISON_MUSHROOM)


def forage_for_season(season: int) -> Optional[ForageKind]:
    """The mountain item that season's tiles give, or ``None`` (Winter)."""
    return FORAGE_BY_SEASON.get(int(season))


def forage_for_name(name: str) -> ForageKind:
    for kind in ALL_KINDS:
        if kind.name == name:
            return kind
    raise KeyError(f"unknown forage kind: {name}")


def loop_hours(kind: ForageKind) -> int:
    """Whole in-game hours one loop takes (the task refuses a next loop that cannot land by 16:00)."""
    return -(-int(kind.loop_frames) // TICKS_PER_HOUR)


def loops_before_bin_closes(hour: int, minute: int, loop_frames: int) -> int:
    """How many house -> bin loops of ``loop_frames`` fit before 17:00 from this clock time.

    ``minute`` is the ROM minute byte (0-14, 60 ticks each): the clock is 900 ticks an hour.
    """
    now = int(hour) * TICKS_PER_HOUR + int(minute) * 60
    left = BIN_CLOSE_HOUR * TICKS_PER_HOUR - now - BIN_MARGIN_FRAMES
    return max(0, left // max(1, int(loop_frames)))


def trips_today(kind: ForageKind, hour: int, minute: int = 0) -> int:
    """Trips worth planning from this clock time: loops that fit, capped at ``max_per_day``."""
    return min(kind.max_per_day, loops_before_bin_closes(hour, minute, kind.loop_frames))


def forage_first_today(
    *,
    kind: Optional[ForageKind],
    hour: int,
    has_harvest: bool,
    late_day: bool,
    is_post_storm: bool = False,
    is_festival: bool = False,
) -> bool:
    """True when the trip goes ahead of crop work: nothing ripe to ship, so the morning is free.

    Watering, sowing and clearing carry on after 18:00, but a ripe crop pays only if it reaches
    the bin before 17:00, so a harvest day keeps its morning for the harvest.
    """
    return bool(
        FORAGE_FIRST
        and kind is not None
        and not has_harvest
        and not late_day
        and not is_post_storm
        and not is_festival
        and int(hour) <= FORAGE_FIRST_LATEST_HOUR
        and trips_today(kind, hour) > 0
    )


__all__ = [
    "ForageKind",
    "GRAPE",
    "GREEN_FRUIT",
    "MUSHROOM",
    "ALL_KINDS",
    "FORAGE_BY_SEASON",
    "FORAGE_FIRST",
    "POISON_MUSHROOM",
    "forage_first_today",
    "forage_for_name",
    "forage_for_season",
    "loop_hours",
    "loops_before_bin_closes",
    "trips_today",
]
