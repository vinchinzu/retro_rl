"""Watering-can refill selection for the main F0 pond corridor.

ROM fact (CheckToolSuccess / ToolAnimationWateringCan): farm fill only when
the tile-in-front property is F0/F9–FD (can → 0x14). Prefer the named
``map_config`` main-pond stands over generic water-edge search so west-pocket
plants open one corridor instead of ranking non-fill streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from harvest.maps.map_config import (
    FARM_MAIN_POND_STANDS,
    FARM_POND_ACCESS_STAGING_TILES,
    FARM_POND_REFILL_CORRIDOR,
    farm_pond_refill_stands,
    player_in_west_plant_pocket,
)

PathFn = Callable[
    [Tuple[int, int], Tuple[int, int]],
    Optional[Sequence[Tuple[int, int]]],
]


# Preferred CheckToolSuccess fill properties (raw map IDs that usually map).
REFILL_PREFERRED_WATER_TILES = frozenset({0xF0, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD})
REFILL_NONFILL_WATER_TILES = frozenset({0xF1, 0xF2, 0xF7, 0xF8})


@dataclass(frozen=True)
class RefillTarget:
    """Stand tile + face for a can refill attempt."""

    stand: Tuple[int, int]
    face: str
    source: str  # "main_pond_corridor" | "preferred_edge" | "staging"
    pathable: bool = True


def select_main_pond_refill(
    player: Tuple[int, int],
    find_path: PathFn,
    *,
    bad_stands: Optional[set] = None,
) -> Optional[RefillTarget]:
    """Pick the nearest pathable stand from FARM_MAIN_POND_STANDS.

    ``find_path(start, goal)`` should return a full path list ending at goal,
    or None if unreachable. After a y=31 fence gap open the player is often on
    the *north* lip — preferring a fixed south-lip order then times out walking
    around the pond body.
    """
    blocked = bad_stands or set()
    hits: List[Tuple[int, int, RefillTarget]] = []
    for rank, (stand, face) in enumerate(farm_pond_refill_stands()):
        if stand in blocked:
            continue
        path = find_path(player, stand)
        if path is not None:
            dist = abs(stand[0] - player[0]) + abs(stand[1] - player[1])
            hits.append(
                (
                    dist,
                    rank,
                    RefillTarget(
                        stand=stand,
                        face=face,
                        source="main_pond_corridor",
                        pathable=True,
                    ),
                )
            )
    if not hits:
        return None
    hits.sort(key=lambda row: (row[0], row[1]))
    return hits[0][2]


def select_staging_stand(
    player: Tuple[int, int],
    find_path: PathFn,
    *,
    staging_tiles: Sequence[Tuple[int, int]] = FARM_POND_ACCESS_STAGING_TILES,
) -> Optional[RefillTarget]:
    """Nearest pathable staging tile north of the y=31 fence wall."""
    if player in staging_tiles:
        return RefillTarget(
            stand=player,
            face="down",
            source="staging",
            pathable=True,
        )
    if not player_in_west_plant_pocket(player):
        return None
    ordered = sorted(
        staging_tiles,
        key=lambda t: abs(t[0] - player[0]) + abs(t[1] - player[1]),
    )
    for stage in ordered:
        path = find_path(player, stage)
        if path is not None:
            return RefillTarget(
                stand=stage,
                face="down",
                source="staging",
                pathable=True,
            )
    return None


def corridor_needs_fence_open(
    player: Tuple[int, int],
    find_path: PathFn,
    *,
    blocking_fences: Sequence[Tuple[int, int]],
    bad_stands: Optional[set] = None,
) -> bool:
    """True when main pond is unreachable and the y=31 fence wall is up."""
    if not blocking_fences:
        return False
    if select_main_pond_refill(player, find_path, bad_stands=bad_stands) is not None:
        return False
    # Only force fence-open from the west plant pocket (or just north of wall).
    if player_in_west_plant_pocket(player) or player[1] <= 31:
        return True
    return False


def order_preferred_edges(
    edges: Sequence[Tuple[Tuple[int, int], str]],
    player: Tuple[int, int],
    *,
    water_id_for: Callable[[Tuple[int, int], str], int],
    preferred_ids: frozenset = REFILL_PREFERRED_WATER_TILES,
) -> List[Tuple[Tuple[int, int], str]]:
    """Sort water edges: preferred fill id → main-pond stand rank → distance.

    Main-pond stands from map_config always sort before other preferred water.
    """
    pond_rank = {
        stand: idx for idx, (stand, _face) in enumerate(FARM_MAIN_POND_STANDS)
    }

    def key(edge: Tuple[Tuple[int, int], str]) -> Tuple[int, int, int, int]:
        tile, face = edge
        wid = water_id_for(tile, face)
        pref = 0 if wid in preferred_ids else 1
        if tile in pond_rank:
            band = 0
            rank = pond_rank[tile]
        else:
            band = 1
            rank = 99
        dist = abs(tile[0] - player[0]) + abs(tile[1] - player[1])
        return (pref, band, rank, dist)

    return sorted(edges, key=key)


def is_no_work_reason(reason: Optional[str]) -> bool:
    """True when a crop task SUCCESS is intentional no-op (not crop work)."""
    if not reason:
        return False
    r = reason.strip().lower()
    return r.startswith("no_work") or r.startswith("no_work:")


def crop_completion_status(
    *,
    work_mode: str,
    planted: int,
    watered: int,
    dry_at_start: int,
    refill_exhausted: bool,
    had_seed_stock: bool,
    rain: bool = False,
) -> Tuple[str, str]:
    """Return (status, reason) for CropWaterTask terminal outcome.

    status is ``success`` | ``no_work`` | ``failure``.
    Callers map ``no_work`` → TaskStatus.SUCCESS with a no_work reason so the
    day plan can journal it separately from real crop work.
    """
    mode = (work_mode or "full").strip().lower()
    if rain and mode != "water" and planted == 0 and watered == 0:
        return ("no_work", "no_work: rain; no plant/water work")

    if mode == "water":
        if watered > 0:
            return (
                "success",
                f"planted={planted} watered={watered}",
            )
        if dry_at_start <= 0:
            return ("no_work", "no_work: water-only; no dry crop tiles")
        if refill_exhausted:
            return (
                "failure",
                f"water fail: refill exhausted with dry_crops={dry_at_start}",
            )
        return (
            "failure",
            f"water fail: dry_crops={dry_at_start} watered=0",
        )

    if mode == "establish":
        if planted > 0:
            return ("success", f"planted={planted} watered={watered}")
        if not had_seed_stock:
            return ("no_work", "no_work: no plantable seed stock")
        return ("failure", "establish fail: planted=0 with seed stock")

    # full mode
    if planted > 0 or watered > 0:
        return ("success", f"planted={planted} watered={watered}")
    if dry_at_start > 0 and refill_exhausted:
        return (
            "failure",
            f"crop fail: refill exhausted with dry_crops={dry_at_start}",
        )
    if dry_at_start > 0:
        return ("failure", f"crop fail: dry_crops={dry_at_start} no progress")
    if not had_seed_stock:
        return ("no_work", "no_work: no plots and no plantable seed stock")
    return ("no_work", "no_work: no plots detected")


__all__ = [
    "FARM_POND_REFILL_CORRIDOR",
    "REFILL_NONFILL_WATER_TILES",
    "REFILL_PREFERRED_WATER_TILES",
    "RefillTarget",
    "corridor_needs_fence_open",
    "crop_completion_status",
    "is_no_work_reason",
    "order_preferred_edges",
    "select_main_pond_refill",
    "select_staging_stand",
]
