"""Shared animal and livestock RAM helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.core.ram_catalog import field_spec, live_wram_base, read_animal_slot_field, read_ram_u8, read_ram_u16

ADDR_CHICKEN_COUNT = field_spec("num_chickens").address
ADDR_COW_COUNT = field_spec("num_cows").address
ADDR_NUM_COWS = ADDR_COW_COUNT
ADDR_HAY_COUNT = field_spec("stored_grass").address
ADDR_STORED_GRASS = ADDR_HAY_COUNT
ADDR_FED_CHICKENS_N = field_spec("fed_chickens_n").address
ADDR_FED_CHICKENS_FLAGS = field_spec("fed_chickens_flags").address
ADDR_FED_COWS_N = field_spec("fed_cows_n").address
ADDR_FED_COWS_FLAGS = field_spec("fed_cows_flags").address
ADDR_ITEM_ON_HAND = field_spec("held_item").address
ADDR_HELD_ITEM = ADDR_ITEM_ON_HAND
ADDR_INCUBATOR_FLAGS = field_spec("incubator_flags").address
ADDR_EGG_AVAILABLE = field_spec("egg_available").address

ITEM_EGG = 0x14
ITEM_FODDER = 0x1A
ITEM_CHICKEN_FEED = ITEM_FODDER
CHICKEN_SLOT_BASE = 0xC286
CHICKEN_SLOT_SIZE = 8
CHICKEN_SLOT_COUNT = 12
CHICKEN_SLOT_POSITION_X_OFFSET = 0x04
CHICKEN_SLOT_POSITION_Y_OFFSET = 0x06
TILE_SIZE = 16
INCUBATOR_EGG_TILES = frozenset({(14, 11)})
COW_SLOT_COUNT = 12
INCUBATOR_BIT = 0x2000
COW_DAILY_BRUSHED_FLAG = 0x01
COW_DAILY_MILKED_FLAG = 0x02
COW_DAILY_TALKED_FLAG = 0x04
COW_DAILY_ATTENTION_FLAGS = COW_DAILY_BRUSHED_FLAG | COW_DAILY_TALKED_FLAG
COW_STATUS_BABY_FLAG = 0x02
COW_STATUS_YOUNG_FLAG = 0x04
COW_STATUS_ADULT_FLAG = 0x08
COW_STATUS_CRANKY_FLAG = 0x10
COW_STATUS_SICK_FLAG = 0x20
COW_STATUS_PREGNANT_FLAG = 0x40

# ROM sickness and overnight health mechanics (bank_83.asm:6723 CowFeedingandStatus)
ITEM_COW_MEDICINE = 0x09
COW_MEDICINE_PRICE_G = 1000
COW_SICKNESS_START_TIMER = 7          # Byte +3 initialized to 7 on contracting sickness
COW_UNFED_HAPPINESS_PENALTY = -8      # bank_83.asm:6816 LDA #$FFF8
COW_SICK_HAPPINESS_PENALTY = -12      # bank_83.asm:6875 LDA #$FFF4
COW_UNFED_SICK_OVERNIGHT_DELTA = -20  # Total overnight drop when an unfed cow rolls sick (-8 + -12)
COW_SICK_ROLL_DENOMINATOR = 4         # RNGReturn0toA with Y=4 -> 1 in 4 (25%) chance of sickness

# ROM cranky and outside pasture overnight mechanics (bank_83.asm:6881 CowFeedingandStatus)
COW_CRANKY_START_TIMER = 3              # Byte +3 initialized to 3 on becoming cranky (bank_83.asm:6938 LDA #$03)
COW_CRANKY_HAPPINESS_PENALTY = -8       # bank_83.asm:6902 LDA #$FFF8
COW_CRANKY_OUTSIDE_ROLL_DENOMINATOR = 16 # bank_83.asm:6898 LDA #$10 -> 1 in 16 (6.25%) chance when outdoors with $7F1F5A:1000
WEATHER_WILD_DOG_BREACH_FLAG = 0x0400   # $0196 bit 0x0400 set by broken fence overnight check (bank_82.asm:2499)
EVENT_FLAG_WILD_DOG_BREACH = 0x0200     # $7F1F6E bit 0x0200


def read_hay_count(ram: np.ndarray) -> int:
    """Read 16-bit hay/stored grass count."""
    return read_ram_u16(ram, ADDR_HAY_COUNT)


def read_stored_grass(ram: np.ndarray) -> int:
    return read_hay_count(ram)


def read_item_on_hand(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_ITEM_ON_HAND)


def read_held_item(ram: np.ndarray) -> int:
    return read_item_on_hand(ram)


def read_num_cows(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_COW_COUNT)


def read_fed_chickens_n(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_FED_CHICKENS_N)


def read_fed_chickens_flags(ram: np.ndarray) -> int:
    return read_ram_u16(ram, ADDR_FED_CHICKENS_FLAGS)


def read_fed_cows_n(ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_FED_COWS_N)


def read_fed_cows_flags(ram: np.ndarray) -> int:
    return read_ram_u16(ram, ADDR_FED_COWS_FLAGS)


def read_cow_status(ram: np.ndarray, slot: int) -> int:
    return read_animal_slot_field(ram, "cow", slot, "status_raw")


def read_cow_daily_flags(ram: np.ndarray, slot: int) -> int:
    return read_animal_slot_field(ram, "cow", slot, "raw_1")


def read_cow_happiness(ram: np.ndarray, slot: int) -> int:
    return read_animal_slot_field(ram, "cow", slot, "happiness")


def read_cow_timer(ram: np.ndarray, slot: int) -> int:
    """Read byte +3 (pregnancy countdown 21..0, sick countdown 7..0, or age 0..21)."""
    return read_animal_slot_field(ram, "cow", slot, "pregnancy_raw")


def cow_is_sick(ram: np.ndarray, slot: int) -> bool:
    """True if cow exists and has the sickness bit set (0x20)."""
    status = read_cow_status(ram, slot)
    return bool((status & 0x01) and (status & COW_STATUS_SICK_FLAG))


def sick_cow_slots(ram: np.ndarray) -> list[int]:
    """Extant cow slots that currently have the sick flag set."""
    return [slot for slot in existing_cow_slots(ram) if cow_is_sick(ram, slot)]


def cow_days_until_death(ram: np.ndarray, slot: int) -> Optional[int]:
    """Days left until sick cow dies (decrements 7 down to 0). None if not sick."""
    if not cow_is_sick(ram, slot):
        return None
    return read_cow_timer(ram, slot)


def ram_has_sick_cows(ram: np.ndarray) -> bool:
    """True if any cow on the farm is currently sick."""
    return bool(sick_cow_slots(ram))


def ram_needs_cow_medicine(ram: np.ndarray) -> bool:
    """True if any cow is sick and needs medicine (item 0x09)."""
    return ram_has_sick_cows(ram)


def cow_is_cranky(ram: np.ndarray, slot: int) -> bool:
    """True if cow exists and has the cranky bit set (0x10)."""
    status = read_cow_status(ram, slot)
    return bool((status & 0x01) and (status & COW_STATUS_CRANKY_FLAG))


def cranky_cow_slots(ram: np.ndarray) -> list[int]:
    """Extant cow slots that currently have the cranky flag set."""
    return [slot for slot in existing_cow_slots(ram) if cow_is_cranky(ram, slot)]


def read_cow_cranky_timer(ram: np.ndarray, slot: int) -> Optional[int]:
    """Countdown days remaining for cranky cow (3 down to 0). None if not cranky."""
    if not cow_is_cranky(ram, slot):
        return None
    return read_cow_timer(ram, slot)


def ram_has_cranky_cows(ram: np.ndarray) -> bool:
    """True if any cow on the farm is currently cranky."""
    return bool(cranky_cow_slots(ram))


def cow_slot_exists(ram: np.ndarray, slot: int) -> bool:
    return bool(read_cow_status(ram, slot) & 0x01)


def existing_cow_slots(ram: np.ndarray) -> list[int]:
    slots: list[int] = []
    for slot in range(COW_SLOT_COUNT):
        try:
            if cow_slot_exists(ram, slot):
                slots.append(slot)
        except Exception:
            break
    return slots


# CowFeedingandStatus ($83BC5A): a barn adult tests fed_cows_flags at its slot
# bit, a pregnant one at 0x1000. A clear bit costs happiness -8 and a sick roll;
# a sick cow dies seven nights later.
COW_BARN_MAP = 0x27
PREGNANT_COW_FEED_FLAG = 0x1000


def cows_outside(ram: np.ndarray) -> list[int]:
    """Cows that walked out of the barn door: the ROM stores the season number as their home map.

    A cow outside gets no talk, brush or milk from the barn chores, and rain costs her happiness at night.
    """
    return [slot for slot in existing_cow_slots(ram) if read_animal_slot_field(ram, "cow", slot, "home_map_raw") < 4]


def unfed_adult_cow_slots(ram: np.ndarray) -> list[int]:
    """Barn adults the night pass will count as unfed."""
    fed = read_fed_cows_flags(ram)
    unfed: list[int] = []
    for slot in existing_cow_slots(ram):
        status = read_cow_status(ram, slot)
        if not status & COW_STATUS_ADULT_FLAG:
            continue
        if read_animal_slot_field(ram, "cow", slot, "home_map_raw") != COW_BARN_MAP:
            continue
        mask = PREGNANT_COW_FEED_FLAG if status & COW_STATUS_PREGNANT_FLAG else 1 << slot
        if not fed & mask:
            unfed.append(slot)
    return unfed


def cow_needs_daily_attention(ram: np.ndarray, slot: int) -> bool:
    flags = read_cow_daily_flags(ram, slot)
    return (flags & COW_DAILY_ATTENTION_FLAGS) != COW_DAILY_ATTENTION_FLAGS


def cow_is_milk_ready(ram: np.ndarray, slot: int) -> bool:
    status = read_cow_status(ram, slot)
    if not (status & COW_STATUS_ADULT_FLAG):
        return False
    # The milk script refuses status bits 0x70 ($849E0B): cranky, sick, pregnant.
    return not bool(status & (COW_STATUS_CRANKY_FLAG | COW_STATUS_SICK_FLAG | COW_STATUS_PREGNANT_FLAG))


def cow_needs_milking(ram: np.ndarray, slot: int) -> bool:
    return cow_is_milk_ready(ram, slot) and not bool(read_cow_daily_flags(ram, slot) & COW_DAILY_MILKED_FLAG)


def is_holding_egg(ram: np.ndarray) -> bool:
    """True when the player is currently holding an egg."""
    return read_item_on_hand(ram) == ITEM_EGG


def is_incubating(ram: np.ndarray) -> bool:
    """True when an egg is currently in the incubator."""
    return bool(read_ram_u16(ram, ADDR_INCUBATOR_FLAGS) & INCUBATOR_BIT)


def egg_available_today(ram: np.ndarray) -> bool:
    """True when the chicken has laid an egg that has not been picked up yet."""
    return read_egg_available_flags(ram) > 0


def read_egg_available_flags(ram: np.ndarray) -> int:
    """Read the bitfield of floor eggs available in the coop."""
    return read_ram_u16(ram, ADDR_EGG_AVAILABLE)


def count_chicken_slots(ram: np.ndarray) -> Tuple[int, int, int]:
    """Count (adults, chicks, eggs) across the 12 chicken slots."""
    adults = chicks = eggs = 0
    base = live_wram_base(ram)
    for i in range(CHICKEN_SLOT_COUNT):
        addr = base + CHICKEN_SLOT_BASE + i * CHICKEN_SLOT_SIZE
        if addr >= len(ram):
            break
        b = int(ram[addr])
        if not (b & 0x01):
            continue
        age = (b >> 1) & 0x07
        if age >= 4:
            adults += 1
        elif age >= 2:
            chicks += 1
        else:
            eggs += 1
    return adults, chicks, eggs


def chicken_slot_eggs_available(ram: np.ndarray) -> bool:
    """True when persistent chicken slots show an uncollected egg object."""
    base = live_wram_base(ram)
    incubating = bool(read_ram_u16(ram, ADDR_INCUBATOR_FLAGS) & INCUBATOR_BIT)
    for i in range(CHICKEN_SLOT_COUNT):
        addr = base + CHICKEN_SLOT_BASE + i * CHICKEN_SLOT_SIZE
        if addr >= len(ram):
            break
        status = int(ram[addr])
        if not (status & 0x01):
            continue
        age = (status >> 1) & 0x07
        if age >= 2:
            continue
        x_addr = addr + CHICKEN_SLOT_POSITION_X_OFFSET
        y_addr = addr + CHICKEN_SLOT_POSITION_Y_OFFSET
        if y_addr + 1 >= len(ram):
            return True
        x = int(ram[x_addr]) | (int(ram[x_addr + 1]) << 8)
        y = int(ram[y_addr]) | (int(ram[y_addr + 1]) << 8)
        tile = (x // TILE_SIZE, y // TILE_SIZE)
        if incubating and tile in INCUBATOR_EGG_TILES:
            continue
        return True
    return False


def count_cow_slots(ram: np.ndarray) -> int:
    return len(existing_cow_slots(ram))


def ram_has_chickens(ram: np.ndarray) -> bool:
    """Return True when live/save RAM has at least one chicken."""
    return read_ram_u8(ram, ADDR_CHICKEN_COUNT) > 0


def ram_has_cows(ram: np.ndarray) -> bool:
    """Return True when live/save RAM has at least one cow."""
    return read_ram_u8(ram, ADDR_COW_COUNT) > 0


def ram_needs_chicken_chores(ram: np.ndarray) -> bool:
    """Return True when any chicken chore remains for the current day."""
    adults, chicks, eggs = count_chicken_slots(ram)
    if adults <= 0 and chicks == 0 and eggs == 0:
        adults = read_ram_u8(ram, ADDR_CHICKEN_COUNT)
    if adults <= 0:
        return egg_available_today(ram)
    fed = read_fed_chickens_n(ram)
    needs_feed = fed < adults and (
        read_hay_count(ram) > 0 or read_ram_u8(ram, ADDR_ITEM_ON_HAND) == ITEM_CHICKEN_FEED
    )
    return needs_feed or egg_available_today(ram) or chicken_slot_eggs_available(ram) or is_holding_egg(ram)


def ram_needs_cow_chores(ram: np.ndarray) -> bool:
    """Return True when any cow still needs feed, talk, brushing, or medicine today."""
    slots = existing_cow_slots(ram)
    cows = max(read_ram_u8(ram, ADDR_COW_COUNT), len(slots))
    if cows <= 0:
        return read_ram_u8(ram, ADDR_ITEM_ON_HAND) == ITEM_FODDER
    has_fodder = read_hay_count(ram) > 0 or read_ram_u8(ram, ADDR_ITEM_ON_HAND) == ITEM_FODDER
    unfed = unfed_adult_cow_slots(ram)
    needs_feed = (bool(unfed) or read_ram_u8(ram, ADDR_FED_COWS_N) < cows) and has_fodder
    needs_attention = any(cow_needs_daily_attention(ram, slot) for slot in slots)
    needs_milk = any(cow_needs_milking(ram, slot) for slot in slots)
    needs_medicine = any(cow_is_sick(ram, slot) for slot in slots)
    return needs_feed or needs_attention or needs_milk or needs_medicine


__all__ = [
    "ADDR_CHICKEN_COUNT",
    "ADDR_COW_COUNT",
    "ADDR_NUM_COWS",
    "ADDR_HAY_COUNT",
    "ADDR_STORED_GRASS",
    "ADDR_FED_CHICKENS_N",
    "ADDR_FED_CHICKENS_FLAGS",
    "ADDR_FED_COWS_N",
    "ADDR_FED_COWS_FLAGS",
    "ADDR_ITEM_ON_HAND",
    "ADDR_HELD_ITEM",
    "ADDR_INCUBATOR_FLAGS",
    "ADDR_EGG_AVAILABLE",
    "ITEM_EGG",
    "ITEM_FODDER",
    "ITEM_CHICKEN_FEED",
    "ITEM_COW_MEDICINE",
    "COW_MEDICINE_PRICE_G",
    "COW_SICKNESS_START_TIMER",
    "COW_UNFED_HAPPINESS_PENALTY",
    "COW_SICK_HAPPINESS_PENALTY",
    "COW_UNFED_SICK_OVERNIGHT_DELTA",
    "COW_SICK_ROLL_DENOMINATOR",
    "COW_CRANKY_START_TIMER",
    "COW_CRANKY_HAPPINESS_PENALTY",
    "COW_CRANKY_OUTSIDE_ROLL_DENOMINATOR",
    "WEATHER_WILD_DOG_BREACH_FLAG",
    "EVENT_FLAG_WILD_DOG_BREACH",
    "CHICKEN_SLOT_BASE",
    "CHICKEN_SLOT_SIZE",
    "CHICKEN_SLOT_COUNT",
    "INCUBATOR_EGG_TILES",
    "COW_SLOT_COUNT",
    "INCUBATOR_BIT",
    "COW_DAILY_BRUSHED_FLAG",
    "COW_DAILY_MILKED_FLAG",
    "COW_DAILY_TALKED_FLAG",
    "COW_DAILY_ATTENTION_FLAGS",
    "COW_STATUS_BABY_FLAG",
    "COW_STATUS_YOUNG_FLAG",
    "COW_STATUS_ADULT_FLAG",
    "COW_STATUS_CRANKY_FLAG",
    "COW_STATUS_SICK_FLAG",
    "COW_STATUS_PREGNANT_FLAG",
    "read_hay_count",
    "read_stored_grass",
    "read_item_on_hand",
    "read_held_item",
    "read_num_cows",
    "read_fed_chickens_n",
    "read_fed_chickens_flags",
    "read_fed_cows_n",
    "read_fed_cows_flags",
    "read_cow_status",
    "read_cow_daily_flags",
    "read_cow_happiness",
    "read_cow_timer",
    "cow_is_sick",
    "sick_cow_slots",
    "cow_days_until_death",
    "ram_has_sick_cows",
    "ram_needs_cow_medicine",
    "cow_is_cranky",
    "cranky_cow_slots",
    "read_cow_cranky_timer",
    "ram_has_cranky_cows",
    "cow_slot_exists",
    "existing_cow_slots",
    "unfed_adult_cow_slots",
    "cow_needs_daily_attention",
    "cow_is_milk_ready",
    "cow_needs_milking",
    "is_holding_egg",
    "is_incubating",
    "egg_available_today",
    "read_egg_available_flags",
    "chicken_slot_eggs_available",
    "count_chicken_slots",
    "count_cow_slots",
    "ram_has_chickens",
    "ram_has_cows",
    "ram_needs_chicken_chores",
    "ram_needs_cow_chores",
]
