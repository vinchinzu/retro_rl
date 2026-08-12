"""Unified RAM/state fact reader for day-plan decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.day_plan_status import (
    _read_state_ram,
    count_chicken_slots,
    is_rainy_weather,
    ram_has_any_crop_seeds,
    ram_has_farm_debris,
    ram_has_harvestable_crops,
    ram_has_waterable_crops,
    ram_needs_chicken_chores,
    ram_needs_cow_chores,
    ram_should_buy_cow,
    read_world_day_time,
    read_world_weekday,
    state_has_farm_debris,
    state_has_harvestable_crops,
)

StateRamLoader = Callable[[Optional[str]], Optional[np.ndarray]]


@dataclass
class WorldProbe:
    """Read planning facts from live RAM and/or a save-state snapshot.

    When ``ram`` is set, live RAM wins for location-sensitive reads such as
    ``tilemap()``. Crop and animal facts fall back to ``state_name`` when live
    RAM cannot see off-viewport farm tiles.
    """

    ram: Optional[np.ndarray] = None
    state_name: Optional[str] = None
    load_state_ram: StateRamLoader = field(default=_read_state_ram, repr=False)
    _state_ram: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _state_loaded: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_inputs(
        cls,
        *,
        ram: Optional[np.ndarray] = None,
        state_name: Optional[str] = None,
        load_state_ram: StateRamLoader = _read_state_ram,
    ) -> WorldProbe:
        """Build a probe from whichever inputs the caller has."""
        if ram is not None:
            return cls(ram=ram, state_name=state_name, load_state_ram=load_state_ram)
        return cls(state_name=state_name, load_state_ram=load_state_ram)

    @property
    def state_ram(self) -> Optional[np.ndarray]:
        if not self._state_loaded:
            self._state_ram = self.load_state_ram(self.state_name)
            self._state_loaded = True
        return self._state_ram

    @property
    def source_ram(self) -> Optional[np.ndarray]:
        return self.ram if self.ram is not None else self.state_ram

    def _require_ram(self) -> Optional[np.ndarray]:
        return self.source_ram

    def day_time(self) -> Tuple[int, int, int]:
        ram = self._require_ram()
        if ram is None:
            return 0, 6, 0
        return read_world_day_time(ram)

    def calendar_date(self) -> Tuple[int, int]:
        """Return ``(season, day)`` from live/save RAM."""
        from harvest.planner.day_plan_status import read_world_date

        ram = self._require_ram()
        if ram is None:
            return 0, 1
        return read_world_date(ram)

    def weekday(self) -> Optional[int]:
        ram = self._require_ram()
        if ram is None:
            return None
        return read_world_weekday(ram)

    def tilemap(self) -> Optional[int]:
        """Player map id from live RAM when available, else the save snapshot."""
        ram = self.ram if self.ram is not None else self.state_ram
        if ram is None or ADDR_TILEMAP >= len(ram):
            return None
        return int(ram[ADDR_TILEMAP])

    def is_rainy(self) -> bool:
        ram = self._require_ram()
        return False if ram is None else is_rainy_weather(ram)

    def has_any_crop_seeds(self) -> bool:
        ram = self._require_ram()
        return False if ram is None else ram_has_any_crop_seeds(ram)

    def has_seasonal_plantable_seeds(self) -> bool:
        """True when inventory has seeds plantable for today's calendar."""
        from harvest.planner.day_plan_status import ram_has_seasonal_crop_seeds

        ram = self._require_ram()
        if ram is None:
            return False
        season, day = self.calendar_date()
        return ram_has_seasonal_crop_seeds(ram, season, day)

    def resolve_seed_type(self) -> Optional[str]:
        """Pick today's crop seed type from calendar + inventory + shipped."""
        from harvest.planner.day_plan_status import resolve_seed_type_from_ram

        ram = self._require_ram()
        if ram is None:
            return None
        return resolve_seed_type_from_ram(ram)

    def has_waterable_crops(self) -> bool:
        ram = self._require_ram()
        return False if ram is None else ram_has_waterable_crops(
            ram, state_name=self.state_name
        )

    def has_harvestable_crops(self) -> bool:
        if self.ram is not None and ram_has_harvestable_crops(self.ram):
            return True
        if self.state_name is not None:
            return state_has_harvestable_crops(self.state_name)
        return False

    def has_farm_debris(self) -> bool:
        """True when the farm still has clearable weeds/stones/rocks/stumps."""
        if self.ram is not None and ram_has_farm_debris(self.ram):
            return True
        if self.state_name is not None:
            return state_has_farm_debris(self.state_name)
        return False

    def chicken_counts(self) -> Tuple[int, int, int]:
        ram = self._require_ram()
        return (0, 0, 0) if ram is None else count_chicken_slots(ram)

    def needs_chicken_chores(self) -> bool:
        ram = self._require_ram()
        return False if ram is None else ram_needs_chicken_chores(ram)

    def needs_cow_chores(self) -> bool:
        ram = self._require_ram()
        return False if ram is None else ram_needs_cow_chores(ram)

    def should_buy_cow(self) -> bool:
        ram = self._require_ram()
        return False if ram is None else ram_should_buy_cow(ram)

    def money(self) -> Optional[int]:
        """Wallet gold, or None when no RAM is available."""
        from harvest.core.ram_catalog import read_ram_value

        ram = self._require_ram()
        if ram is None:
            return None
        try:
            return int(read_ram_value(ram, "money"))
        except Exception:
            return None


__all__ = ["StateRamLoader", "WorldProbe"]
