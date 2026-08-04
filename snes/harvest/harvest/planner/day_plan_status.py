"""RAM-backed day planning status helpers for Harvest Moon."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.paths import TASKS_DIR as PROJECT_TASKS_DIR
from harvest.tasks.crop_planter import DEFAULT_CROP_BOUNDS, tile_needs_watering
from harvest.core.tile_catalog import CLEARABLE_DEBRIS_TYPES
from harvest.tasks.farm_clearer import ADDR_TILEMAP, TileScanner, get_tile_at
from harvest.tasks.harvest_task import live_harvestable_crop_tiles, state_harvestable_crop_tiles
from harvest.core.animal_status import (
    ADDR_CHICKEN_COUNT,
    ADDR_COW_COUNT,
    ADDR_HAY_COUNT,
    ADDR_FED_CHICKENS_N,
    ADDR_FED_COWS_N,
    ADDR_ITEM_ON_HAND,
    ADDR_INCUBATOR_FLAGS,
    ADDR_EGG_AVAILABLE,
    ITEM_EGG,
    ITEM_FODDER,
    CHICKEN_SLOT_BASE,
    CHICKEN_SLOT_SIZE,
    CHICKEN_SLOT_COUNT,
    INCUBATOR_BIT,
    count_chicken_slots,
    egg_available_today,
    is_holding_egg,
    is_incubating,
    ram_has_chickens,
    ram_has_cows,
    ram_needs_chicken_chores,
    ram_needs_cow_chores,
    read_hay_count,
)
from harvest.core.ram_catalog import field_spec, read_ram_u8, read_ram_u16, read_ram_value
from harvest.runtime.rom_tools import parse_save_state, resolve_state_path

TASKS_DIR = str(PROJECT_TASKS_DIR)
FARM_TILEMAP = 0x00
FARM_TILEMAPS = frozenset({0x00, 0x01, 0x02, 0x03})
HOUSE_TILEMAP = 0x15
# House interiors advance with remodels. Current saves use 0x16 after the
# first expansion; keep 0x17 registered for the coming second expansion so task
# logic does not silently treat it as an unknown building.
HOUSE_TILEMAPS = frozenset({0x15, 0x16, 0x17})
SHED_TILEMAP = 0x26
BARN_TILEMAP = 0x27
COOP_TILEMAP = 0x28


def is_farm_tilemap(tilemap: int) -> bool:
    """True for the outdoor farm map in any season."""
    return int(tilemap) in FARM_TILEMAPS


def tilemaps_match(actual: int, expected: int) -> bool:
    """Compare tilemaps while treating seasonal farm maps as equivalent."""
    actual = int(actual)
    expected = int(expected)
    if actual == expected:
        return True
    return is_farm_tilemap(actual) and is_farm_tilemap(expected)


ADDR_WEEKDAY = field_spec("weekday").address
ADDR_DAY = field_spec("day").address
ADDR_HOUR = field_spec("hour").address
ADDR_MINUTE = field_spec("minute").address
ADDR_SEASON = field_spec("season").address
ADDR_WEATHER = field_spec("weather").address
ADDR_WEATHER_FLAGS = 0x0196
ADDR_TOOL_SELECTED = field_spec("tool_selected").address
ADDR_TOOL_BACKPACK = field_spec("tool_backpack").address
ADDR_CORN_SEEDS = field_spec("corn_seeds").address
ADDR_TOMATO_SEEDS = field_spec("tomato_seeds").address
ADDR_POTATO_SEEDS = field_spec("potato_seeds").address
ADDR_TURNIP_SEEDS = field_spec("turnip_seeds").address
SEED_COUNT_ADDRS = (
    ADDR_CORN_SEEDS,
    ADDR_TOMATO_SEEDS,
    ADDR_POTATO_SEEDS,
    ADDR_TURNIP_SEEDS,
)
SEED_COUNT_ADDR_BY_TYPE = {
    "corn": ADDR_CORN_SEEDS,
    "tomato": ADDR_TOMATO_SEEDS,
    "potato": ADDR_POTATO_SEEDS,
    "turnip": ADDR_TURNIP_SEEDS,
}
ADDR_MONEY = field_spec("money").address
SUNDAY_WEEKDAY = 0
COW_PURCHASE_COST = 5000
RAINY_WEATHER_CODES = frozenset({1, 2, 3})
RAINY_WEATHER_FLAG_MASK = 0x0002 | 0x0008 | 0x0010


def _read_state_ram(state_name: Optional[str]) -> Optional[np.ndarray]:
    """Load save-state RAM once for auto-plan decisions."""
    if not state_name:
        return None
    try:
        state = parse_save_state(resolve_state_path(state_name))
    except FileNotFoundError:
        return None
    return state.ram


def read_state_weekday(state_name: Optional[str]) -> Optional[int]:
    """Read weekday directly from a save-state RAM snapshot."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return None
    return read_ram_u8(ram, ADDR_WEEKDAY)


def read_state_time(state_name: Optional[str]) -> Optional[Tuple[int, int]]:
    """Read hour/minute directly from a save-state RAM snapshot."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return None
    return read_ram_u8(ram, ADDR_HOUR), read_ram_u8(ram, ADDR_MINUTE)


def read_world_day_time(ram: np.ndarray) -> Tuple[int, int, int]:
    """Read live/save-state day+time from either direct WRAM or live WRAM+0x4000."""
    return (
        read_ram_u8(ram, ADDR_DAY),
        read_ram_u8(ram, ADDR_HOUR),
        read_ram_u8(ram, ADDR_MINUTE),
    )


def read_world_date(ram: np.ndarray) -> Tuple[int, int]:
    """Read live/save-state season+day from either direct WRAM or live WRAM+0x4000."""
    return (
        read_ram_u8(ram, ADDR_SEASON),
        read_ram_u8(ram, ADDR_DAY),
    )


def read_world_weekday(ram: np.ndarray) -> int:
    """Read live/save-state weekday from either direct WRAM or live WRAM+0x4000."""
    return read_ram_u8(ram, ADDR_WEEKDAY)


def is_rainy_weather(ram: np.ndarray) -> bool:
    """Return True when current weather means crops do not need manual watering."""
    flags = read_ram_u16(ram, ADDR_WEATHER_FLAGS, live_offset=False)
    return bool(flags & RAINY_WEATHER_FLAG_MASK)


def state_is_rainy(state_name: Optional[str]) -> bool:
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return is_rainy_weather(ram)


def ram_has_waterable_crops(
    ram: np.ndarray,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
    state_name: Optional[str] = None,
) -> bool:
    """Return True when the current visible farm map contains dry/planted crop tiles."""
    left, top, right, bottom = bounds
    skip_tiles = set(live_harvestable_crop_tiles(ram, state_name, bounds=bounds)) if state_name else set()
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if (x, y) in skip_tiles:
                continue
            if tile_needs_watering(get_tile_at(ram, x, y)):
                return True
    return False


def state_has_any_crop_seeds(state_name: Optional[str]) -> bool:
    """Return True when the save already has crop seeds on hand."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return any(addr < len(ram) and int(ram[addr]) > 0 for addr in SEED_COUNT_ADDRS)


def state_has_waterable_crops(state_name: Optional[str]) -> bool:
    """Return True when the save already has visible tiles that need watering."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return ram_has_waterable_crops(ram, state_name=state_name)


def state_has_harvestable_crops(state_name: Optional[str]) -> bool:
    """Return True when the farm already has mature crop tiles ready for harvest."""
    return bool(state_harvestable_crop_tiles(state_name, bounds=DEFAULT_CROP_BOUNDS))


def ram_has_harvestable_crops(ram: np.ndarray, bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS) -> bool:
    """Return True when the live visible farm map contains likely-ripe crop tiles."""
    return bool(live_harvestable_crop_tiles(ram, None, bounds=bounds))


def ram_has_farm_debris(
    ram: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> bool:
    """Return True when clearable weeds/stones/rocks/stumps remain."""
    return TileScanner().has_clearable_debris(ram, bounds)


def state_has_farm_debris(state_name: Optional[str]) -> bool:
    """Return True when the save's farm metatile grid still has debris."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return ram_has_farm_debris(ram)


def state_has_chickens(state_name: Optional[str]) -> bool:
    """Return True when the save has at least one chicken."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return read_ram_u8(ram, ADDR_CHICKEN_COUNT) > 0


def state_chicken_counts(state_name: Optional[str]) -> Tuple[int, int, int]:
    """Return adult/chick/egg chicken-slot counts for a save state."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return 0, 0, 0
    return count_chicken_slots(ram)


def state_needs_chicken_chores(state_name: Optional[str]) -> bool:
    """Return True when the save has unfinished chicken chores today."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return ram_needs_chicken_chores(ram)


def state_has_cows(state_name: Optional[str]) -> bool:
    """Return True when the save has at least one cow."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return read_ram_u8(ram, ADDR_COW_COUNT) > 0


def ram_should_buy_cow(ram: np.ndarray) -> bool:
    """Return True when the route should prioritize buying the first cow."""
    if read_ram_u8(ram, ADDR_COW_COUNT) > 0:
        return False
    if read_ram_value(ram, "money") < COW_PURCHASE_COST:
        return False
    return read_world_weekday(ram) != SUNDAY_WEEKDAY


def state_should_buy_cow(state_name: Optional[str]) -> bool:
    """Return True when a save is ready for the first cow purchase route."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return ram_should_buy_cow(ram)


def state_needs_cow_chores(state_name: Optional[str]) -> bool:
    """Return True when the save has unfinished cow feeding today."""
    ram = _read_state_ram(state_name)
    if ram is None:
        return False
    return ram_needs_cow_chores(ram)


def ram_has_any_crop_seeds(ram: np.ndarray) -> bool:
    """Return True when the live/save RAM already has crop seeds on hand."""
    return any(read_ram_u8(ram, addr) > 0 for addr in SEED_COUNT_ADDRS)


def ram_seed_inventory(ram: np.ndarray) -> dict[str, int]:
    """Return seed counts keyed by crop name."""
    return {
        seed_type: read_ram_u8(ram, addr)
        for seed_type, addr in SEED_COUNT_ADDR_BY_TYPE.items()
    }


def ram_shipped_crop_totals(ram: np.ndarray) -> dict[str, int]:
    """Return shipped crop totals used by ending / ranch-master scoring."""
    from harvest.core.ram_catalog import read_ram_value

    return {
        "turnip": int(read_ram_value(ram, "shipped_turnips")),
        "potato": int(read_ram_value(ram, "shipped_potatoes")),
        "tomato": int(read_ram_value(ram, "shipped_tomatoes")),
        "corn": int(read_ram_value(ram, "shipped_corn")),
    }


def ram_has_seasonal_crop_seeds(
    ram: np.ndarray,
    season: int,
    day: int,
) -> bool:
    """True when inventory has seeds plantable in this season/day."""
    from harvest.planner.crop_planner import resolve_seed_type_for_date

    seed = resolve_seed_type_for_date(
        season,
        day,
        inventory=ram_seed_inventory(ram),
    )
    if seed is None:
        return False
    return ram_seed_count(ram, seed) > 0


def resolve_seed_type_from_ram(ram: np.ndarray) -> str | None:
    """Pick today's seed type from calendar + inventory + shipped totals."""
    from harvest.planner.crop_planner import resolve_seed_type_for_date

    season, day = read_world_date(ram)
    return resolve_seed_type_for_date(
        season,
        day,
        inventory=ram_seed_inventory(ram),
        shipped=ram_shipped_crop_totals(ram),
    )


def ram_seed_count(ram: np.ndarray, seed_type: str = "potato") -> int:
    """Return stored seed count for one crop seed type."""
    addr = SEED_COUNT_ADDR_BY_TYPE.get(seed_type)
    if addr is None:
        return 0
    return read_ram_u8(ram, addr)


def is_house_tilemap(tilemap: int) -> bool:
    """Return True for any farmhouse interior map, including remodel levels."""
    return tilemap in HOUSE_TILEMAPS


__all__ = [
    "TASKS_DIR",
    "FARM_TILEMAP",
    "FARM_TILEMAPS",
    "HOUSE_TILEMAP",
    "HOUSE_TILEMAPS",
    "SHED_TILEMAP",
    "BARN_TILEMAP",
    "COOP_TILEMAP",
    "ADDR_WEEKDAY",
    "ADDR_DAY",
    "ADDR_HOUR",
    "ADDR_MINUTE",
    "ADDR_SEASON",
    "ADDR_WEATHER",
    "ADDR_WEATHER_FLAGS",
    "ADDR_TOOL_SELECTED",
    "ADDR_TOOL_BACKPACK",
    "ADDR_CORN_SEEDS",
    "ADDR_TOMATO_SEEDS",
    "ADDR_POTATO_SEEDS",
    "ADDR_TURNIP_SEEDS",
    "SEED_COUNT_ADDRS",
    "SEED_COUNT_ADDR_BY_TYPE",
    "ADDR_CHICKEN_COUNT",
    "ADDR_COW_COUNT",
    "ADDR_MONEY",
    "ADDR_HAY_COUNT",
    "ADDR_FED_CHICKENS_N",
    "ADDR_FED_COWS_N",
    "ADDR_ITEM_ON_HAND",
    "ADDR_INCUBATOR_FLAGS",
    "ADDR_EGG_AVAILABLE",
    "ITEM_EGG",
    "ITEM_FODDER",
    "CHICKEN_SLOT_BASE",
    "CHICKEN_SLOT_SIZE",
    "CHICKEN_SLOT_COUNT",
    "INCUBATOR_BIT",
    "SUNDAY_WEEKDAY",
    "COW_PURCHASE_COST",
    "RAINY_WEATHER_CODES",
    "RAINY_WEATHER_FLAG_MASK",
    "parse_save_state",
    "resolve_state_path",
    "DEFAULT_CROP_BOUNDS",
    "tile_needs_watering",
    "live_harvestable_crop_tiles",
    "state_harvestable_crop_tiles",
    "read_state_weekday",
    "read_state_time",
    "read_world_day_time",
    "read_world_date",
    "read_world_weekday",
    "is_rainy_weather",
    "state_is_rainy",
    "ram_has_waterable_crops",
    "state_has_any_crop_seeds",
    "state_has_waterable_crops",
    "state_has_harvestable_crops",
    "ram_has_harvestable_crops",
    "ram_has_farm_debris",
    "state_has_farm_debris",
    "CLEARABLE_DEBRIS_TYPES",
    "state_has_chickens",
    "state_chicken_counts",
    "state_needs_chicken_chores",
    "state_has_cows",
    "ram_should_buy_cow",
    "state_should_buy_cow",
    "state_needs_cow_chores",
    "ram_has_any_crop_seeds",
    "ram_has_seasonal_crop_seeds",
    "ram_seed_count",
    "ram_seed_inventory",
    "ram_shipped_crop_totals",
    "resolve_seed_type_from_ram",
    "is_farm_tilemap",
    "is_house_tilemap",
    "tilemaps_match",
    "ram_has_chickens",
    "ram_has_cows",
    "ram_needs_chicken_chores",
    "ram_needs_cow_chores",
    "read_hay_count",
    "is_holding_egg",
    "is_incubating",
    "egg_available_today",
    "count_chicken_slots",
]
