"""D2 leftover smash as four farm quadrants.

Whole-farm stones / rocks / stumps stall on a last distant cell (south-stream
stone at (12,55), SE boulder at (60,51), FA-east hug). Each chunk is a
bounded scan; the four chain to an empty farm. Inclusive tile bounds on the
64×64 metatile grid.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from harvest.core.tile_catalog import MAP_HEIGHT, MAP_WIDTH
from harvest.tasks.farm_clear_quota import ClearQuota, DebrisCounts, quota_counts_met


FARM_CHUNK_ORDER: tuple[str, ...] = ("nw", "ne", "sw", "se")
FARM_CHUNK_BOUNDS: dict[str, tuple[int, int, int, int]] = {
    "nw": (0, 0, 31, 31),
    "ne": (32, 0, 63, 31),
    "sw": (0, 32, 31, 63),
    "se": (32, 32, 63, 63),
}
SMASH_SECTIONS: tuple[str, ...] = ("stones", "rocks", "stumps")
EXHAUSTIVE = 10_000

# Live leftover stalls that the 2×2 split isolates.
CHUNK_PIN_TILES: dict[str, tuple[int, int]] = {
    "nw": (11, 29),   # west plant pocket stone
    "ne": (48, 13),   # FA-east hug
    "sw": (12, 55),   # last on-map stone / south-stream
    "se": (60, 51),   # last boulder
}


def chunk_of_tile(tx: int, ty: int) -> str:
    """Quadrant containing this metatile. Tiles on the split go east / south."""
    east = tx >= 32
    south = ty >= 32
    if south and east:
        return "se"
    if south:
        return "sw"
    if east:
        return "ne"
    return "nw"


def resolve_chunks(chunk: str | Sequence[str] | None = "all") -> tuple[str, ...]:
    """``all`` / None → four quadrants; one name → that chunk only."""
    if chunk is None or chunk == "all":
        return FARM_CHUNK_ORDER
    if isinstance(chunk, str):
        if chunk not in FARM_CHUNK_BOUNDS:
            raise ValueError(f"unknown farm chunk {chunk!r}")
        return (chunk,)
    names = tuple(chunk)
    unknown = [name for name in names if name not in FARM_CHUNK_BOUNDS]
    if unknown:
        raise ValueError(f"unknown farm chunk {unknown[0]!r}")
    return names


def chunk_bounds(chunk: str) -> tuple[int, int, int, int]:
    return FARM_CHUNK_BOUNDS[chunk]


def iter_chunk_bounds(
    chunk: str | Sequence[str] | None = "all",
) -> Iterable[tuple[str, tuple[int, int, int, int]]]:
    for name in resolve_chunks(chunk):
        yield name, FARM_CHUNK_BOUNDS[name]


def chunks_cover_farm() -> bool:
    """True when the four bounds partition the 64×64 farm with no gap."""
    covered = set()
    for _name, (x0, y0, x1, y1) in iter_chunk_bounds("all"):
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                if (tx, ty) in covered:
                    return False
                covered.add((tx, ty))
    return covered == {
        (tx, ty) for ty in range(MAP_HEIGHT) for tx in range(MAP_WIDTH)
    }


def smash_is_clear(counts: DebrisCounts) -> bool:
    """True when no stones, large 2×2, or stumps remain (fences/weeds ignored)."""
    return counts.stones <= 0 and counts.large_rocks <= 0 and counts.stumps <= 0


def wanted_quota(section: str) -> ClearQuota:
    """Day 2 work contract. Oversized counts cap to whatever the pin spawned."""
    if section == "bushes":
        return ClearQuota(weeds=EXHAUSTIVE)
    if section == "fences":
        return ClearQuota(fences=EXHAUSTIVE)
    if section == "stones":
        return ClearQuota(stones=EXHAUSTIVE)
    if section == "rocks":
        return ClearQuota(large_rocks=EXHAUSTIVE)
    if section == "stumps":
        return ClearQuota(stumps=EXHAUSTIVE)
    return ClearQuota(
        weeds=EXHAUSTIVE,
        stones=EXHAUSTIVE,
        large_rocks=EXHAUSTIVE,
        stumps=EXHAUSTIVE,
        fences=EXHAUSTIVE,
    )


def section_complete(
    section: str,
    start: DebrisCounts,
    end: DebrisCounts,
) -> bool:
    """True when this pass removes the section quota from its own start counts.

    Callers that run one chunk must pass start/end already scanned in that
    chunk's bounds. Whole-farm ``all`` uses unbounded counts.
    """
    return quota_counts_met(start, end, wanted_quota(section))


def smash_done_empty(section: str) -> tuple[str, ...]:
    """Glance ``require_empty`` keys when a smash section (or all) is done."""
    if section == "bushes":
        return ("weeds",)
    if section == "fences":
        return ("fences",)
    if section == "stones":
        return ("stones",)
    if section == "rocks":
        return ("large_rocks",)
    if section == "stumps":
        return ("stumps",)
    if section == "all":
        return ("weeds", "fences", "stones", "large_rocks", "stumps")
    return ()


__all__ = [
    "CHUNK_PIN_TILES",
    "EXHAUSTIVE",
    "FARM_CHUNK_BOUNDS",
    "FARM_CHUNK_ORDER",
    "SMASH_SECTIONS",
    "chunk_bounds",
    "chunk_of_tile",
    "chunks_cover_farm",
    "iter_chunk_bounds",
    "resolve_chunks",
    "section_complete",
    "smash_done_empty",
    "smash_is_clear",
    "wanted_quota",
]
