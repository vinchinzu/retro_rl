"""Cure a sick barn cow: buy medicine at the animal shop if none is owned, carry it, dose the cow.

A sick adult dies when its byte +3 (seven nights) reaches zero, so a 1,000G dose saves a 5,000G cow
and 25 happiness points. The dose is a trip of its own (animal shop, then the shed pickup, then the
barn), run before the day's chores; the ordinary chore pass then refetches the brush and milker
because the dose empties one slot of the carry pair.

Measured on the ROM: the buy, the shed delivery and the pickup (``test_care_shopping_rom``), and the
cure itself (``test_herd_ladder_rom``). Not measured: whether rain or snow refuses the delivery the
way it refuses an animal's, so weather does not gate the buy here.
"""

from __future__ import annotations

from typing import List

import numpy as np

from harvest.core.animal_status import (
    COW_MEDICINE_PRICE_G,
    COW_STATUS_ADULT_FLAG,
    read_cow_status,
    read_cow_timer,
    sick_cow_slots,
)
from harvest.core.ram_catalog import read_ram_value
from harvest.planner.day_phase_cow import (
    COW_CURE_PHASE,
    ENTER_BARN_PHASE,
    EXIT_BARN_PHASE,
    NAV_TO_BARN_PHASE,
)
from harvest.planner.day_phase_types import PhaseSpec

MEDICINE_TOOL_ID = 9
# A town trip that leaves later than this lands past the animal shop's hours (it shuts at 17:00).
LAST_BUY_HOUR = 15
# With this many nights or fewer left on the death timer the cure outranks the day's other town trip.
URGENT_NIGHTS = 2

ENSURE_MEDICINE_PHASE = PhaseSpec(
    "ENSURE_MEDICINE", "ensure_tool", {"tool_id": MEDICINE_TOOL_ID}, failure_policy="optional"
)
BUY_MEDICINE_PHASE = PhaseSpec(
    "BUY_MEDICINE",
    "shop_buy",
    {"tool_ids": [MEDICINE_TOOL_ID], "timeout": 16000, "nav_timeout": 6000},
    failure_policy="optional",
)


def sick_adult_slots(ram: np.ndarray) -> List[int]:
    """Sick cows the ROM's medicine script accepts (it wants an adult)."""
    return [slot for slot in sick_cow_slots(ram) if read_cow_status(ram, slot) & COW_STATUS_ADULT_FLAG]


def nights_left(ram: np.ndarray) -> int:
    """Nights before the first sick cow dies (byte +3 counts down from seven)."""
    slots = sick_adult_slots(ram)
    return min((int(read_cow_timer(ram, slot)) for slot in slots), default=99)


def cure_day_phases(ram: np.ndarray, *, other_town_trip: bool = False) -> List[PhaseSpec]:
    """Phases that dose every sick adult today, or ``[]`` when there is nothing to do or no way yet.

    The list starts on the farm: the caller has already planned the exit from the house or barn.

    ``other_town_trip`` is true when an animal or tool purchase already owns today's trip into town:
    a second trip fails or lands after closing, so the buy waits for tomorrow unless the cow is about
    to die. An owned dose (on the shed shelf or in the carry pair) needs no trip at all.
    """
    from harvest.calendar_schedule import shop_open
    from harvest.tasks.buy_seeds import tool_owned

    if not sick_adult_slots(ram):
        return []
    barn = [ENSURE_MEDICINE_PHASE, NAV_TO_BARN_PHASE, ENTER_BARN_PHASE, COW_CURE_PHASE, EXIT_BARN_PHASE]
    if tool_owned(ram, MEDICINE_TOOL_ID):
        return barn
    money = int(read_ram_value(ram, "money"))
    hour = int(read_ram_value(ram, "hour"))
    open_now = shop_open(
        "animal",
        weekday=int(read_ram_value(ram, "weekday")),
        hour=hour,
        season=int(read_ram_value(ram, "season")),
        day=int(read_ram_value(ram, "day")),
        weather_flags=int(read_ram_value(ram, "weather")),
    )
    if money < COW_MEDICINE_PRICE_G or hour >= LAST_BUY_HOUR or not open_now:
        return []
    if other_town_trip and nights_left(ram) > URGENT_NIGHTS:
        return []
    return [BUY_MEDICINE_PHASE, *barn]


__all__ = [
    "BUY_MEDICINE_PHASE",
    "ENSURE_MEDICINE_PHASE",
    "LAST_BUY_HOUR",
    "MEDICINE_TOOL_ID",
    "URGENT_NIGHTS",
    "cure_day_phases",
    "nights_left",
    "sick_adult_slots",
]
