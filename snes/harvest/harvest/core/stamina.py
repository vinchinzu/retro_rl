"""RAM-backed stamina object for scripts, clear, and spa.

Decomp: ``!current_stamina`` $7E0918, ``!max_stamina`` $7E0917,
``!exaustion_level`` $7E096C, hammer/axe hit counter $7E096D
(``CMP.B #$06`` then the 2×2 rock/stump breaks). Hammer/axe
``ChangeStamina #$FE`` is −2 per registered swing.

A large rock needs **6 registered hits**. Tool Y-holds miss, so clear
will not *start* a multi-hit unless stamina covers an **8-swing**
budget (6 + 2 misses) = 16. Mid-rock swings still stop at stam < 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from harvest.core.ram_catalog import (
    LIVE_RAM_WRAM_OFFSET,
    WRAM_SNAPSHOT_SIZE,
    field_spec,
)

SWING_STAMINA_COST = 2
ROM_MULTI_HITS = 6
MULTI_HIT_SWING_BUDGET = 8
LIFT_STAMINA_COST = 1


def _u8_at(ram: np.ndarray, addr: int, offset: int = 0) -> int:
    idx = addr + offset
    if 0 <= idx < len(ram):
        return int(ram[idx])
    return 0


def _stamina_view_offset(ram: np.ndarray) -> int:
    """Pick WRAM vs live +0x4000 from max_stamina, not buffer length alone.

    Task RAM (FarmClearer / HotSpring) is direct WRAM even when the buffer
    is 0x24000. ``harvest_bot world`` snapshots of live ``get_ram()`` store
    WRAM at +0x4000. Prefer the view whose max_stamina is actually set.
    """
    addr = field_spec("max_stamina").address
    direct = _u8_at(ram, addr, 0)
    if len(ram) <= WRAM_SNAPSHOT_SIZE:
        return 0
    live = _u8_at(ram, addr, LIVE_RAM_WRAM_OFFSET)
    if live > 0:
        return LIVE_RAM_WRAM_OFFSET
    if direct > 0:
        return 0
    return LIVE_RAM_WRAM_OFFSET


def _read_u8(ram: np.ndarray, key: str, offset: int | None = None) -> int:
    spec = field_spec(key)
    view = _stamina_view_offset(ram) if offset is None else offset
    return _u8_at(ram, spec.address, view)


def swings_to_finish_multi_hit(tool_hits: int = 0) -> int:
    """Y-holds still needed to finish the current 2×2, including miss budget."""
    remaining = max(0, ROM_MULTI_HITS - int(tool_hits))
    extra = max(0, MULTI_HIT_SWING_BUDGET - ROM_MULTI_HITS)
    return remaining + extra


def stamina_cost_for_hits(required_hits: int, *, tool_hits: int = 0) -> int:
    if int(required_hits) <= 1:
        return SWING_STAMINA_COST
    return swings_to_finish_multi_hit(tool_hits) * SWING_STAMINA_COST


@dataclass(frozen=True)
class Stamina:
    """Live stamina as one object: current/max plus the multi-hit counter."""

    current: int
    maximum: int
    exhaustion: int = 0
    tool_hits: int = 0

    @classmethod
    def from_ram(cls, ram: np.ndarray) -> "Stamina":
        offset = _stamina_view_offset(ram)
        current = _read_u8(ram, "stamina", offset)
        maximum = _read_u8(ram, "max_stamina", offset)
        if maximum <= 0:
            maximum = 100
        return cls(
            current=current,
            maximum=maximum,
            exhaustion=_read_u8(ram, "exhaustion_level", offset),
            tool_hits=_read_u8(ram, "tool_hit_counter", offset),
        )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Stamina":
        current = int(row.get("current", row.get("stamina", 0)) or 0)
        maximum = int(row.get("maximum", row.get("max_stamina", 100)) or 100)
        if maximum <= 0:
            maximum = 100
        return cls(
            current=current,
            maximum=maximum,
            exhaustion=int(row.get("exhaustion", 0) or 0),
            tool_hits=int(row.get("tool_hits", 0) or 0),
        )

    @property
    def is_full(self) -> bool:
        return self.current >= self.maximum

    @property
    def deficit(self) -> int:
        return max(0, self.maximum - self.current)

    def can_afford_swings(self, swings: int) -> bool:
        return self.current >= int(swings) * SWING_STAMINA_COST

    def can_lift(self) -> bool:
        return self.current >= LIFT_STAMINA_COST

    def can_finish_multi_hit(self, *, tool_hits: int | None = None) -> bool:
        hits = self.tool_hits if tool_hits is None else int(tool_hits)
        return self.can_afford_swings(swings_to_finish_multi_hit(hits))

    def cost_to_clear(self, required_hits: int, *, lifting: bool = False) -> int:
        if lifting:
            return LIFT_STAMINA_COST
        return stamina_cost_for_hits(required_hits, tool_hits=self.tool_hits)

    def can_afford_clear(self, required_hits: int, *, lifting: bool = False) -> bool:
        return self.current >= self.cost_to_clear(required_hits, lifting=lifting)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": int(self.current),
            "maximum": int(self.maximum),
            "exhaustion": int(self.exhaustion),
            "tool_hits": int(self.tool_hits),
            "is_full": self.is_full,
            "deficit": self.deficit,
            "can_finish_rock": self.can_finish_multi_hit(),
            "rock_swing_budget": swings_to_finish_multi_hit(self.tool_hits),
            "rock_stamina_need": stamina_cost_for_hits(
                ROM_MULTI_HITS, tool_hits=self.tool_hits
            ),
        }

    def __int__(self) -> int:
        return int(self.current)

    def __index__(self) -> int:
        return int(self.current)

    def __bool__(self) -> bool:
        return self.current > 0

    def _as_int(self, other: object) -> int | None:
        if isinstance(other, Stamina):
            return other.current
        if isinstance(other, (int, np.integer)):
            return int(other)
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Stamina):
            return (
                self.current == other.current
                and self.maximum == other.maximum
                and self.exhaustion == other.exhaustion
                and self.tool_hits == other.tool_hits
            )
        as_int = self._as_int(other)
        if as_int is None:
            return NotImplemented
        return self.current == as_int

    def __lt__(self, other: object) -> bool:
        as_int = self._as_int(other)
        if as_int is None:
            return NotImplemented
        return self.current < as_int

    def __le__(self, other: object) -> bool:
        as_int = self._as_int(other)
        if as_int is None:
            return NotImplemented
        return self.current <= as_int

    def __gt__(self, other: object) -> bool:
        as_int = self._as_int(other)
        if as_int is None:
            return NotImplemented
        return self.current > as_int

    def __ge__(self, other: object) -> bool:
        as_int = self._as_int(other)
        if as_int is None:
            return NotImplemented
        return self.current >= as_int

    def __str__(self) -> str:
        return f"{self.current}/{self.maximum}"

    def __repr__(self) -> str:
        return (
            f"Stamina({self.current}/{self.maximum} "
            f"hits={self.tool_hits} exh={self.exhaustion})"
        )


__all__ = [
    "LIFT_STAMINA_COST",
    "MULTI_HIT_SWING_BUDGET",
    "ROM_MULTI_HITS",
    "SWING_STAMINA_COST",
    "Stamina",
    "stamina_cost_for_hits",
    "swings_to_finish_multi_hit",
]
