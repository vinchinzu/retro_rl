"""Coop interior layout constants for CoopChoresTask.

Extracted from ``coop_task`` so chore FSM composition stays thin.
Constants and pure stand helpers here must stay free of task state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from harvest.maps.map_config import find_landmark
from harvest.tasks.nav import TILE_SIZE

# ── Coop interior layout (tilemap 0x28) ─────────────────────────
# Positions discovered via coop_chores / coop_sell_egg recording traces.

FEED_BIN_STAND: Tuple[int, int] = (2, 6)
FEED_BIN_FACE: str = "left"
FEED_CLEAR_STAND: Tuple[int, int] = (2, 3)
VISIBLE_EGG_SPRITE = 0x00F3


@dataclass(frozen=True)
class ChickenFeedSpot:
    stand: Tuple[int, int]
    face: str
    interact_px: Tuple[int, int]
    flag: int


CHICKEN_FEED_FLAGS: Tuple[int, ...] = (
    0x0001,
    0x0002,
    0x0004,
    0x0008,
    0x0010,
    0x0020,
    0x0040,
    0x0080,
    0x0100,
    0x0200,
    0x0400,
    0x0800,
)

# Coop feed trough tiles are the top row tile properties 0xE2..0xED.
# The reliable interaction point, from recordings, is near the lower-left
# of the stand tile while holding Up+A into the trough.
CHICKEN_FEED_SPOTS: Tuple[ChickenFeedSpot, ...] = tuple(
    ChickenFeedSpot(
        stand=(x, 3),
        face="up",
        interact_px=(x * TILE_SIZE + (6 if x == 2 else 10), 3 * TILE_SIZE + 14),
        flag=flag,
    )
    for x, flag in zip(range(2, 14), CHICKEN_FEED_FLAGS)
)

EGG_PICKUP_STAND: Tuple[int, int] = (2, 4)
EGG_PICKUP_FACE: str = "left"
CHICKEN_EGG_SPAWN_PIXELS: Tuple[Tuple[int, int], ...] = (
    (0x18, 0x48),
    (0x38, 0x58),
    (0x48, 0x98),
    (0x58, 0x78),
    (0x68, 0xA8),
    (0x78, 0x88),
    (0x88, 0x58),
    (0x98, 0x98),
    (0xA8, 0x78),
    (0xB8, 0xA8),
    (0xC8, 0x68),
    (0xD8, 0x88),
    (0x28, 0xA8),
)
CHICKEN_EGG_FLAGS: Tuple[int, ...] = tuple(1 << slot for slot in range(len(CHICKEN_EGG_SPAWN_PIXELS)))


def egg_recording_stand(px: int, py: int) -> Tuple[Tuple[int, int], str]:
    """Stand tile/face for a spawn pixel, avoiding the false-open x=5 column."""
    tx, ty = px // TILE_SIZE, py // TILE_SIZE
    stand = (tx + 1, ty)
    if stand[0] == 5:
        return (tx, ty + 1), "up"
    return stand, "left"


EGG_PICKUP_SPOTS: Tuple[Tuple[int, Tuple[int, int], str], ...] = (
    (0x02, (4, 5), "left"),
    (0x01, (2, 4), "left"),
    *(
        (flag, *egg_recording_stand(px, py))
        for flag, (px, py) in zip(CHICKEN_EGG_FLAGS[2:], CHICKEN_EGG_SPAWN_PIXELS[2:])
    ),
)

COOP_ENTRY_STAND: Tuple[int, int] = (8, 12)
COOP_MAIN_AISLE_TOP: Tuple[int, int] = (8, 6)
COOP_LEFT_TOP_APPROACH: Tuple[int, int] = (4, 5)

INCUBATOR_STAND: Tuple[int, int] = (13, 11)
INCUBATOR_FACE: str = "right"
INCUBATOR_APPROACH: Tuple[Tuple[int, int], ...] = ((8, 10), (10, 11), INCUBATOR_STAND)

_EGG_SHIPPING_BIN_LANDMARK = find_landmark("egg_shipping_bin", tilemap_id=0x28)

# Bottom-left coop shipping bin is interacted with from the aisle tile just
# above the bin frontage, facing down into the bin.
SHIP_BIN_STAND: Tuple[int, int] = (2, 10)
SHIP_BIN_INTERACT_STAND: Tuple[int, int] = (
    _EGG_SHIPPING_BIN_LANDMARK[1].tile
    if _EGG_SHIPPING_BIN_LANDMARK is not None
    else (1, 10)
)
SHIP_BIN_FACE: str = (
    _EGG_SHIPPING_BIN_LANDMARK[1].face
    if _EGG_SHIPPING_BIN_LANDMARK is not None and _EGG_SHIPPING_BIN_LANDMARK[1].face
    else "down"
)
SHIP_LANE_X = 38
SHIP_APPROACH_Y = 165
SHIP_INTERACT_PX = (22, 169)
SHIP_RIGHT_LANE_CORNER: Tuple[int, int] = (3, 10)

# Stage at the same door tile EXIT_COOP expects. Handing off at the left
# service lane (3,11) left EXIT_COOP to cross the false-open x=5 edge and time
# out (seen at pos=(57,184) tile=(3,11)).
EXIT_PREP_STAND: Tuple[int, int] = COOP_ENTRY_STAND
# The coop tilemap reports a walkable vertical strip around x=5 that is not
# actually passable. Long-run stalls consistently pin at (5, 11) / (86, 183).
COOP_FALSE_OPEN_COLUMN_X = 5
COOP_FALSE_OPEN_MIN_Y = 8
EXIT_PREP_ESCAPE_ROUTE: Tuple[Tuple[int, int], ...] = (
    (5, 12),
    (2, 12),
    EXIT_PREP_STAND,
)
EXIT_PREP_LEFT_ROUTE: Tuple[Tuple[int, int], ...] = (
    (2, 12),
    EXIT_PREP_STAND,
)
MAX_EXIT_PREP_FRAMES = 360
MAX_EGG_NAV_FRAMES = 480
MAX_EGG_ATTEMPTS = 4
MAX_EGG_DEFERRALS = 1
MAX_FEED_PLACE_FRAMES = 480
MAX_FEED_SLOT_DEFERRALS = 1
MAX_FLOCK_SIZE = 12
# F3 night loops. An egg at timer 0 hatches on the night the byte reaches 3,
# and only when chicks_N is still below 12. A chick reaches adult at 7.
EGG_HATCH_NIGHTS = 3
CHICK_ADULT_NIGHTS = 7
# A shop buy stores 12 into the held feed counter. It does not add.
CHICKEN_FEED_STACK = 12
# CODE_83C0E5 strips an outdoor adult (map < 4) when these weather bits are set.
# The following night CODE_83C05E deletes a status of exactly exists ($01).
CHICKEN_STORM_MASK = 0x0410
CHICKEN_COOP_MAP = 0x28
CHICKEN_ADULT_STATUS = 0x09


def flock_must_stay_in(weather_flags: int) -> bool:
    """Hurricane (and bit $0400) will strip an adult left on a map below 4."""
    return (int(weather_flags) & CHICKEN_STORM_MASK) != 0


def egg_route(
    mode: str,
    *,
    chicks_n: int,
    slot_total: int,
    incubating: bool,
) -> str:
    """Where a held egg goes. At chicks_N 12, or a full slot total, ship it.

    Gift still carries the egg out. An occupied incubator cannot take another.
    """
    if mode == "gift":
        return "gift"
    if mode == "ship":
        return "ship"
    full = int(chicks_n) >= MAX_FLOCK_SIZE or int(slot_total) >= MAX_FLOCK_SIZE
    if mode == "incubate" and not full:
        return "incubate"
    if mode == "auto" and not incubating and not full:
        return "incubate"
    return "ship"


def flock_drop_alarm(before: int, after: int) -> str | None:
    if int(after) < int(before):
        return f"flock drop {int(before)}->{int(after)}"
    return None


def flock_sale_allowed(count: int, target: int = MAX_FLOCK_SIZE) -> bool:
    """Selling is only the oversupply path. The target flock is kept."""
    return int(count) > int(target)


def chicken_feed_short(stock: int, hens: int) -> bool:
    return int(hens) > 0 and int(stock) < int(hens)


def hen_unfed_days(hay: int, hens: int, days: int) -> int:
    """Mornings that cannot place one feed unit per hen. Hay stacks; the
    shop bundle of 12 does not.
    """
    stock = int(hay)
    unfed = 0
    for _ in range(int(days)):
        if stock < hens:
            unfed += 1
        else:
            stock -= hens
    return unfed


def chicken_night_status(map_byte: int, status: int, weather_flags: int, *, fed: bool) -> int:
    """Adult night. Coop + fed stays. A storm outdoors stores status $01."""
    if not (status & 0x01) or not (status & 0x08):
        return int(status)
    if int(map_byte) == CHICKEN_COOP_MAP and fed:
        return int(status)
    if int(map_byte) < 4 and (int(weather_flags) & CHICKEN_STORM_MASK):
        return 0x01
    if int(map_byte) == CHICKEN_COOP_MAP or int(map_byte) >= 4:
        return int(status) | 0x10
    return int(status)


def status_is_dead_chicken(status: int) -> bool:
    return bool(status & 0x01) and not bool(status & 0x0E)


def flock_after_storm(
    *,
    indoor: bool,
    weather_flags: int = 0x0010,
    nights: int = 2,
    count: int = MAX_FLOCK_SIZE,
    fed: bool = True,
) -> int:
    """Two nights: the storm writes $01, and the next night deletes it."""
    status = CHICKEN_ADULT_STATUS
    map_byte = CHICKEN_COOP_MAP if indoor else 0
    alive = int(count)
    for _ in range(int(nights)):
        if alive <= 0:
            break
        if status_is_dead_chicken(status):
            alive -= 1
            status = 0
            continue
        status = chicken_night_status(map_byte, status, weather_flags, fed=fed)
    return alive


def grow_flock(start_adults: int, nights: int) -> int:
    """Incubate one egg at a time from ``start_adults`` fed hens.

    The egg is placed before sleep. The hatch night does not also increment
    the new chick's timer. chicks_N stops at 12 and a further egg stays put.
    """
    adults = int(start_adults)
    chicks_n = adults
    egg_timer: int | None = None
    chick_timers: list[int] = []
    for _ in range(int(nights)):
        if egg_timer is None and adults >= 1 and chicks_n < MAX_FLOCK_SIZE:
            egg_timer = 0
        hatched: list[int] = []
        if egg_timer is not None:
            if egg_timer < EGG_HATCH_NIGHTS:
                egg_timer += 1
            if egg_timer >= EGG_HATCH_NIGHTS and chicks_n < MAX_FLOCK_SIZE:
                egg_timer = None
                chicks_n += 1
                hatched.append(0)
            elif egg_timer >= EGG_HATCH_NIGHTS:
                egg_timer = EGG_HATCH_NIGHTS
        matured: list[int] = []
        for timer in chick_timers:
            timer += 1
            if timer >= CHICK_ADULT_NIGHTS:
                adults += 1
            else:
                matured.append(timer)
        chick_timers = matured + hatched
    return chicks_n


def flock_growth_matches_f3(morning_counts: list[int]) -> bool:
    """True when morning chicks_N rises by one on the 3-night hatch clock.

    A sick or travel prefix may stay flat. The first three rises are 3
    mornings apart, the first of them no earlier than the third morning,
    and the series ends at 4 or more without passing 12.
    """
    counts = [int(count) for count in morning_counts]
    if len(counts) < 2 or counts[0] < 1 or counts[-1] < 4:
        return False
    if any(count < 1 or count > MAX_FLOCK_SIZE for count in counts):
        return False
    rises: list[int] = []
    for index in range(1, len(counts)):
        delta = counts[index] - counts[index - 1]
        if delta < 0 or delta > 1:
            return False
        if delta == 1:
            rises.append(index)
    if len(rises) < 3 or rises[0] < EGG_HATCH_NIGHTS:
        return False
    return (
        rises[1] - rises[0] == EGG_HATCH_NIGHTS
        and rises[2] - rises[1] == EGG_HATCH_NIGHTS
    )
