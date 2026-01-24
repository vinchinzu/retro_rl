"""Shipper window detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Default hourly window for pickup (heuristic).
DEFAULT_PICKUP_START = 17  # 5 PM
DEFAULT_PICKUP_END = 18    # 6 PM


@dataclass
class ShipperWindow:
    start_hour: int = DEFAULT_PICKUP_START
    end_hour: int = DEFAULT_PICKUP_END

    def is_open(self, hour: int) -> bool:
        return self.start_hour <= hour < self.end_hour

    def is_closed(self, hour: int) -> bool:
        return hour >= self.end_hour or hour < self.start_hour


@dataclass
class ShipperWindowDetector:
    """Detects whether shipping can still be added for the day.

    Two modes:
      - RAM: if shipping bin flag is exposed in RAM (preferred)
      - TIME: fallback to hour-based heuristic
    """

    # TODO: wire to actual RAM flag (HM-Decomp suggests a "shipping bin" bit)
    shipping_bin_flag_addr: Optional[int] = None
    flag_mask: int = 0x01
    window: ShipperWindow = ShipperWindow()

    def from_ram(self, ram: np.ndarray) -> Optional[bool]:
        if self.shipping_bin_flag_addr is None:
            return None
        if self.shipping_bin_flag_addr >= len(ram):
            return None
        flag = int(ram[self.shipping_bin_flag_addr])
        return (flag & self.flag_mask) == 0

    def from_time(self, hour: int) -> bool:
        return self.window.is_open(hour)

    def can_ship(self, ram: Optional[np.ndarray], hour: int) -> bool:
        if ram is not None:
            via_ram = self.from_ram(ram)
            if via_ram is not None:
                return via_ram
        return self.from_time(hour)
