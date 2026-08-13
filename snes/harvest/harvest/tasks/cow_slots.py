"""Slot selection, care needs, and feed-goal helpers for CowChoresTask."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.core.animal_probe import cow_slot_snapshots
from harvest.core.animal_status import (
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_TALKED_FLAG,
    cow_needs_milking,
    existing_cow_slots,
    read_cow_daily_flags,
    read_fed_cows_flags,
    read_fed_cows_n,
)
from harvest.tasks.cow_geometry import (
    COW_FEED_SPOTS,
    COW_TALK_FACE,
    COW_TALK_STAND,
    CowFeedSpot,
    count_fed_trough_flags,
    facing_tile,
    feed_route_for_spot,
    next_unfed_spot,
)


class CowSlotsMixin:
    """Barn slot queries and feed accounting."""

    def _facing_tile(self, stand: Tuple[int, int], face: str) -> Tuple[int, int]:
        return facing_tile(stand, face)

    def _select_target_cow_slot(self, ram: np.ndarray) -> Optional[int]:
        rows = cow_slot_snapshots(ram, require_barn=True)
        if not rows:
            slots = existing_cow_slots(ram)
            return slots[0] if slots else None

        target_tile = facing_tile(COW_TALK_STAND, COW_TALK_FACE)
        for row in rows:
            tile = row.get("tile")
            if isinstance(tile, list) and tuple(tile) == target_tile:
                return int(row["slot"])

        def score(row: dict[str, object]) -> int:
            tile = row.get("tile")
            if not isinstance(tile, list) or len(tile) != 2:
                return 999
            return abs(int(tile[0]) - target_tile[0]) + abs(int(tile[1]) - target_tile[1])

        return int(min(rows, key=score)["slot"])

    def _cow_flag_set(self, ram: np.ndarray, flag: int) -> bool:
        if self._target_cow_slot is None:
            return False
        return bool(read_cow_daily_flags(ram, self._target_cow_slot) & flag)

    def _cow_flag_set_for_slot(self, ram: np.ndarray, slot: int, flag: int) -> bool:
        return bool(read_cow_daily_flags(ram, slot) & flag)

    def _milkable_cow_slots(self, ram: np.ndarray) -> list[int]:
        return [
            slot
            for slot in existing_cow_slots(ram)
            if cow_needs_milking(ram, slot) and slot not in self._skipped_milk_slots
        ]

    def _barn_cow_slots(self, ram: np.ndarray) -> list[int]:
        rows = cow_slot_snapshots(ram, require_barn=True)
        slots = [int(row["slot"]) for row in rows if "slot" in row]
        return slots or existing_cow_slots(ram)

    def _slot_needs_talk(self, ram: np.ndarray, slot: int) -> bool:
        return (
            self.talk
            and slot not in self._skipped_talk_slots
            and not self._cow_flag_set_for_slot(ram, slot, COW_DAILY_TALKED_FLAG)
        )

    def _slot_needs_brush(self, ram: np.ndarray, slot: int) -> bool:
        return (
            self.brush
            and self._brush_in_carry_pair(ram)
            and slot not in self._skipped_brush_slots
            and not self._cow_flag_set_for_slot(ram, slot, COW_DAILY_BRUSHED_FLAG)
        )

    def _slot_ready_for_milk(self, ram: np.ndarray, slot: int) -> bool:
        return True

    def _slot_needs_milk(self, ram: np.ndarray, slot: int) -> bool:
        return (
            self.milk
            and self._milker_in_carry_pair(ram)
            and slot not in self._skipped_milk_slots
            and cow_needs_milking(ram, slot)
            and self._slot_ready_for_milk(ram, slot)
        )

    def _slot_needs_care(self, ram: np.ndarray, slot: int) -> bool:
        return (
            self._slot_needs_talk(ram, slot)
            or self._slot_needs_brush(ram, slot)
            or self._slot_needs_milk(ram, slot)
        )

    def _care_needed_cow_slots(self, ram: np.ndarray) -> list[int]:
        return [slot for slot in self._barn_cow_slots(ram) if self._slot_needs_care(ram, slot)]

    def _feedable_cow_slots(self, ram: np.ndarray) -> list[int]:
        return existing_cow_slots(ram)

    def _feed_goal(self, ram: np.ndarray) -> int:
        return min(len(self._feedable_cow_slots(ram)), len(COW_FEED_SPOTS))

    def _current_feed_goal(self, ram: np.ndarray) -> int:
        return self._feed_goal_count or self._feed_goal(ram)

    def _fed_trough_count(self, ram: np.ndarray) -> int:
        return count_fed_trough_flags(read_fed_cows_flags(ram), self._current_feed_goal(ram))

    def _fed_count_now(self, ram: np.ndarray) -> int:
        return max(read_fed_cows_n(ram), self._fed_trough_count(ram))

    def _next_feed_spot(self, ram: np.ndarray) -> CowFeedSpot:
        return next_unfed_spot(read_fed_cows_flags(ram), self._current_feed_goal(ram))

    def _feed_route(self, spot: CowFeedSpot) -> Tuple[Tuple[int, int], ...]:
        return feed_route_for_spot(spot)
