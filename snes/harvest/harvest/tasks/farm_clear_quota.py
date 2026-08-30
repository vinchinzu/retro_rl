"""D2 leftover clear quotas — RAM-count stop, not whole-farm wipe.

``FarmClearTask(handoff="quota", quota=...)`` succeeds when the clearer has
removed at least the requested counts. Small rocks are tile ``0x06``;
large boulders are one 2×2 (TL ``0x0D``/damage). Do not count four cells
of one boulder as four rocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Set

from harvest.core.tile_catalog import (
    FENCE,
    LARGE_ROCK_DAMAGE_TILES,
    LARGE_ROCK_TILES,
    ROCK as SMALL_ROCK_TILE,
    STALE_TILE_IDS,
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
    fences: int = 0

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
            fences=int(data.get("fences", 0) or 0),
        )

    def is_empty(self) -> bool:
        return not any(
            (
                self.weeds,
                self.stones,
                self.small_rocks,
                self.large_rocks,
                self.stumps,
                self.fences,
            )
        )


@dataclass(frozen=True)
class DebrisCounts:
    weeds: int = 0
    stones: int = 0
    small_rocks: int = 0
    large_rocks: int = 0
    stumps: int = 0
    fences: int = 0

    def as_dict(self) -> dict:
        return {
            "weeds": self.weeds,
            "stones": self.stones,
            "small_rocks": self.small_rocks,
            "large_rocks": self.large_rocks,
            "stumps": self.stumps,
            "fences": self.fences,
        }

    def cleared_since(self, now: "DebrisCounts") -> "DebrisCounts":
        return DebrisCounts(
            weeds=self.weeds - now.weeds,
            stones=self.stones - now.stones,
            small_rocks=self.small_rocks - now.small_rocks,
            large_rocks=self.large_rocks - now.large_rocks,
            stumps=self.stumps - now.stumps,
            fences=self.fences - now.fences,
        )

    def meets(self, quota: ClearQuota) -> bool:
        return (
            self.weeds >= quota.weeds
            and self.stones >= quota.stones
            and self.small_rocks >= quota.small_rocks
            and self.large_rocks >= quota.large_rocks
            and self.stumps >= quota.stumps
            and self.fences >= quota.fences
        )


def classify_target(tile_id: int, debris_type: DebrisType) -> str:
    if debris_type == DebrisType.WEED or tile_id == WEED:
        return "weeds"
    if debris_type == DebrisType.STONE or tile_id == STONE:
        return "stones"
    if debris_type == DebrisType.FENCE or tile_id == FENCE:
        return "fences"
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
        "fences": 0,
    }
    for target in targets:
        key = classify_target(int(target.tile_id), target.debris_type)
        if key in tallies:
            tallies[key] += 1
    return DebrisCounts(**tallies)


def capped_quota(want: ClearQuota, start: DebrisCounts) -> ClearQuota:
    """Do not demand more than the pin actually spawned.

    D2 ``Y1_After_Buy_Potato`` has 0× ``0x06`` small boulders. A leftover
    quota of 10 small rocks (pond-tossed ``0x04``) must not fail the
    4-boulder hammer pass.
    """
    return ClearQuota(
        weeds=min(want.weeds, start.weeds),
        stones=min(want.stones, start.stones),
        small_rocks=min(want.small_rocks, start.small_rocks),
        large_rocks=min(want.large_rocks, start.large_rocks),
        stumps=min(want.stumps, start.stumps),
        fences=min(want.fences, start.fences),
    )


def quota_counts_met(
    start: DebrisCounts, now: DebrisCounts, want: ClearQuota
) -> bool:
    if want.is_empty():
        return False
    effective = capped_quota(want, start)
    if effective.is_empty():
        return True
    return start.cleared_since(now).meets(effective)


def unmet_debris_types(
    start: DebrisCounts | None,
    now: DebrisCounts,
    quota: Mapping[str, Any] | ClearQuota | None,
) -> Optional[Set[DebrisType]]:
    """Debris kinds still below the (capped) quota, or None if no quota."""
    want = quota if isinstance(quota, ClearQuota) else ClearQuota.from_mapping(quota)
    if want.is_empty() or not isinstance(start, DebrisCounts):
        return None
    effective = capped_quota(want, start)
    if effective.is_empty():
        return set()
    cleared = start.cleared_since(now)
    unmet: Set[DebrisType] = set()
    if cleared.weeds < effective.weeds:
        unmet.add(DebrisType.WEED)
    if cleared.stones < effective.stones:
        unmet.add(DebrisType.STONE)
    if (
        cleared.small_rocks < effective.small_rocks
        or cleared.large_rocks < effective.large_rocks
    ):
        unmet.add(DebrisType.ROCK)
    if cleared.stumps < effective.stumps:
        unmet.add(DebrisType.STUMP)
    if cleared.fences < effective.fences:
        unmet.add(DebrisType.FENCE)
    return unmet


def farm_map_loaded(ram) -> bool:
    """False on shed-door 0xFF / viewport unload (counts look like a wipe).

    Standing on a8 next to the door still unloads distant metatiles to 0xFF,
    so a player-tile check is not enough.
    """
    from harvest.core.tile_catalog import ADDR_MAP, MAP_WIDTH
    from harvest.tasks.nav import get_pos_from_ram, get_tile_at, TILE_SIZE

    pos = get_pos_from_ram(ram)
    tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
    if int(get_tile_at(ram, *tile)) in STALE_TILE_IDS:
        return False
    end = min(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, len(ram))
    if end <= ADDR_MAP:
        return False
    import numpy as np

    chunk = np.asarray(ram[ADDR_MAP:end], dtype=np.uint8)
    stale = int(np.isin(chunk, list(STALE_TILE_IDS)).sum())
    return stale < 64


def needs_shed_door_step_off(ram) -> bool:
    """True on the shed warp or while the farm map is still unloaded.

    Adjacent a8 is walkable but still unloads distant metatiles — keep
    walking toward (25,28) a1 until :func:`farm_map_loaded`.
    """
    from harvest.tasks.farm_ops import SHED_DOOR_TILE
    from harvest.tasks.nav import get_pos_from_ram, get_tile_at, TILE_SIZE

    pos = get_pos_from_ram(ram)
    tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
    if tile == SHED_DOOR_TILE:
        return True
    if int(get_tile_at(ram, *tile)) in STALE_TILE_IDS:
        return True
    return not farm_map_loaded(ram)


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
    Requested counts cap at what the pin spawned. A capped-empty quota
    (pin spawned none of the requested debris) is a no-op when the farm
    map is loaded, even if ``cleared_count`` is 0. Viewport-unload zeros
    stay False. ``cleared_by_kind`` is only a fallback if no snapshot
    exists.
    """
    want = (
        quota
        if isinstance(quota, ClearQuota)
        else ClearQuota.from_mapping(quota)
    )
    if want.is_empty() or clearer is None:
        return False
    if not farm_map_loaded(ram):
        return False
    start = getattr(clearer, "quota_start_counts", None)
    if isinstance(start, DebrisCounts):
        scan_bounds = bounds
        if scan_bounds is None:
            scan_bounds = getattr(clearer, "farm_bounds", None)
        if not quota_counts_met(start, count_debris(ram, scan_bounds), want):
            return False
        if int(getattr(clearer, "cleared_count", 0) or 0) > 0:
            return True
        # Non-empty effective quota still needs a swing so shed-door unload
        # cannot fake a wipe. Capped-empty is an honest no-op.
        return capped_quota(want, start).is_empty()
    got = getattr(clearer, "cleared_by_kind", None)
    if isinstance(got, Mapping):
        return DebrisCounts(
            weeds=int(got.get("weeds", 0) or 0),
            stones=int(got.get("stones", 0) or 0),
            small_rocks=int(got.get("small_rocks", 0) or 0),
            large_rocks=int(got.get("large_rocks", 0) or 0),
            stumps=int(got.get("stumps", 0) or 0),
            fences=int(got.get("fences", 0) or 0),
        ).meets(want)
    return False


__all__ = [
    "ClearQuota",
    "DebrisCounts",
    "capped_quota",
    "classify_target",
    "count_debris",
    "farm_map_loaded",
    "needs_shed_door_step_off",
    "quota_counts_met",
    "quota_satisfied",
    "unmet_debris_types",
]
