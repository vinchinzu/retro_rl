"""D2 leftover clear quotas — RAM-count stop, not whole-farm wipe.

``FarmClearTask(handoff="quota", quota=...)`` succeeds when the clearer has
removed at least the requested counts. Small rocks are tile ``0x06``;
large boulders are one 2×2 (TL ``0x0D``/damage). Do not count four cells
of one boulder as four rocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from harvest.core.tile_catalog import (
    LARGE_ROCK_DAMAGE_TILES,
    LARGE_ROCK_TILES,
    ROCK as SMALL_ROCK_TILE,
    STONE,
    STUMP_TILES,
    WEED,
    DebrisType,
)
from harvest.tasks.farm_ops import TileScanner


@dataclass(frozen=True)
class ClearQuota:
    weeds: int = 0
    stones: int = 0
    small_rocks: int = 0
    large_rocks: int = 0
    stumps: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ClearQuota":
        if not data:
            return cls()
        return cls(
            weeds=int(data.get("weeds", 0) or 0),
            stones=int(data.get("stones", 0) or 0),
            small_rocks=int(data.get("small_rocks", 0) or 0),
            large_rocks=int(data.get("large_rocks", 0) or 0),
            stumps=int(data.get("stumps", 0) or 0),
        )

    def is_empty(self) -> bool:
        return not any(
            (self.weeds, self.stones, self.small_rocks, self.large_rocks, self.stumps)
        )


@dataclass(frozen=True)
class DebrisCounts:
    weeds: int = 0
    stones: int = 0
    small_rocks: int = 0
    large_rocks: int = 0
    stumps: int = 0

    def as_dict(self) -> dict:
        return {
            "weeds": self.weeds,
            "stones": self.stones,
            "small_rocks": self.small_rocks,
            "large_rocks": self.large_rocks,
            "stumps": self.stumps,
        }

    def cleared_since(self, now: "DebrisCounts") -> "DebrisCounts":
        return DebrisCounts(
            weeds=self.weeds - now.weeds,
            stones=self.stones - now.stones,
            small_rocks=self.small_rocks - now.small_rocks,
            large_rocks=self.large_rocks - now.large_rocks,
            stumps=self.stumps - now.stumps,
        )

    def meets(self, quota: ClearQuota) -> bool:
        return (
            self.weeds >= quota.weeds
            and self.stones >= quota.stones
            and self.small_rocks >= quota.small_rocks
            and self.large_rocks >= quota.large_rocks
            and self.stumps >= quota.stumps
        )


def classify_target(tile_id: int, debris_type: DebrisType) -> str:
    if debris_type == DebrisType.WEED or tile_id == WEED:
        return "weeds"
    if debris_type == DebrisType.STONE or tile_id == STONE:
        return "stones"
    if debris_type == DebrisType.STUMP or tile_id in STUMP_TILES:
        return "stumps"
    if tile_id == SMALL_ROCK_TILE:
        return "small_rocks"
    if tile_id in LARGE_ROCK_TILES or tile_id in LARGE_ROCK_DAMAGE_TILES:
        return "large_rocks"
    if debris_type == DebrisType.ROCK:
        return "large_rocks"
    return "other"


def count_debris(ram, bounds=None, *, types=None) -> DebrisCounts:
    targets = TileScanner().scan(ram, bounds, types=types)
    tallies = {
        "weeds": 0,
        "stones": 0,
        "small_rocks": 0,
        "large_rocks": 0,
        "stumps": 0,
    }
    for target in targets:
        key = classify_target(int(target.tile_id), target.debris_type)
        if key in tallies:
            tallies[key] += 1
    return DebrisCounts(**tallies)


def quota_satisfied(
    ram,
    quota: Mapping[str, Any] | ClearQuota | None,
    *,
    clearer: Optional[Any] = None,
    bounds=None,
) -> bool:
    """True when this pass has cleared at least the requested counts.

    Honest path: ``FarmClearTask.reset`` snapshots ``quota_start_counts``,
    then start-minus-now via ``count_debris`` (one target per 2×2 TL).
    ``cleared_by_kind`` is only a fallback if no snapshot exists.
    """
    want = (
        quota
        if isinstance(quota, ClearQuota)
        else ClearQuota.from_mapping(quota)
    )
    if want.is_empty() or clearer is None:
        return False
    start = getattr(clearer, "quota_start_counts", None)
    if isinstance(start, DebrisCounts):
        scan_bounds = bounds
        if scan_bounds is None:
            scan_bounds = getattr(clearer, "farm_bounds", None)
        return start.cleared_since(count_debris(ram, scan_bounds)).meets(want)
    got = getattr(clearer, "cleared_by_kind", None)
    if isinstance(got, Mapping):
        return DebrisCounts(
            weeds=int(got.get("weeds", 0) or 0),
            stones=int(got.get("stones", 0) or 0),
            small_rocks=int(got.get("small_rocks", 0) or 0),
            large_rocks=int(got.get("large_rocks", 0) or 0),
            stumps=int(got.get("stumps", 0) or 0),
        ).meets(want)
    return False


__all__ = [
    "ClearQuota",
    "DebrisCounts",
    "classify_target",
    "count_debris",
    "quota_satisfied",
]
