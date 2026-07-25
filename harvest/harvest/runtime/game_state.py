"""Live RAM-backed game state helpers for the play session."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from harvest.core.ram_catalog import field_spec, read_ram_value

SEASON_NAMES = {0: "Spring", 1: "Summer", 2: "Fall", 3: "Winter"}
DAY_NAMES = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
ITEM_NAMES = {
    0x00: "Empty",
    0x01: "Sickle", 0x02: "Hoe", 0x03: "Hammer", 0x04: "Axe",
    0x10: "Watering Can",
    0x11: "Gold Sickle", 0x12: "Gold Hoe", 0x13: "Gold Hammer", 0x14: "Gold Axe",
}


class GameState:
    """Parse game state from live RAM.

    env.get_ram() has a 0x4000 header; WRAM address X is at get_ram()[X + 0x4000].
    Game stores money as value/10 in a u24.
    """

    # get_ram() offset: WRAM addr -> get_ram()[addr + OFF]
    OFF = 0x4000
    # WRAM addresses (from save-state / harvest_state.py)
    _YEAR = field_spec("year").address
    _SEASON = field_spec("season").address
    _WEEKDAY = field_spec("weekday").address
    _DAY = field_spec("day").address
    _HOUR = field_spec("hour").address
    _MINUTE = field_spec("minute").address
    _MONEY = field_spec("money").address

    def __init__(self, info: Dict, ram: Optional[np.ndarray] = None):
        self.stamina = info.get('stamina', 0)
        self.item = info.get('item_in_hand', 0)

        if ram is not None:
            self.year = read_ram_value(ram, "year", raw=True)
            self.season = read_ram_value(ram, "season", raw=True)
            self.day_of_week = read_ram_value(ram, "weekday", raw=True)
            self.day = read_ram_value(ram, "day", raw=True)
            self.hour = read_ram_value(ram, "hour", raw=True)
            self.minute = read_ram_value(ram, "minute", raw=True)
            self.money = read_ram_value(ram, "money")
        else:
            self.year = info.get('year', 0)
            self.season = info.get('season', 0)
            self.day_of_week = info.get('day_of_week', 0)
            self.day = info.get('day', 1)
            self.hour = info.get('hour', 6)
            self.minute = info.get('minute', 0)
            self.money = 0

    @property
    def season_name(self) -> str:
        return SEASON_NAMES.get(self.season, "?")

    @property
    def day_name(self) -> str:
        return DAY_NAMES.get(self.day_of_week, "?")

    @property
    def item_name(self) -> str:
        return ITEM_NAMES.get(self.item, f"0x{self.item:02X}")

    @property
    def display_year(self) -> int:
        """Game year (1-based). SRAM stores 0-based."""
        return self.year + 1 if self.year < 10 else self.year

    @property
    def date_str(self) -> str:
        return f"Y{self.display_year} {self.season_name} {self.day} ({self.day_name})"

    @property
    def time_str(self) -> str:
        return f"{self.hour}:00 AM" if self.hour <= 12 else f"{self.hour-12}:00 PM"

    def state_name(self) -> str:
        return f"Y{self.display_year}_{self.season_name}_Day{self.day:02d}_{self.hour:02d}h{self.minute:02d}m"

