"""RAM read helpers and candidate ranking for Waldo cursor/scene discovery."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from retro_harness.ram_state import RamDelta, diff_changed, snapshot


@dataclass(frozen=True)
class CursorCandidate:
    """A RAM address that moved consistently with a cursor axis probe."""

    address: int
    axis: str
    hits: int
    last_before: int
    last_after: int


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned byte from a WRAM snapshot."""
    return int(ram[address])


def filter_byte_range(
    deltas: list[RamDelta],
    *,
    lo: int = 0,
    hi: int = 255,
) -> list[RamDelta]:
    """Keep deltas whose after value is within [lo, hi]."""
    return [d for d in deltas if lo <= d.after <= hi]


def rank_axis_candidates(
    probe_deltas: list[list[RamDelta]],
    *,
    axis: str,
    min_hits: int = 2,
) -> list[CursorCandidate]:
    """Rank addresses that changed in the expected direction across probes.

    Args:
        probe_deltas: One list of RamDelta per movement pulse on this axis.
        axis: Label such as \"x\" or \"y\".
        min_hits: Minimum number of probes an address must appear in.
    """
    counts: Counter[int] = Counter()
    last: dict[int, RamDelta] = {}
    for deltas in probe_deltas:
        seen: set[int] = set()
        for delta in deltas:
            if delta.address in seen:
                continue
            seen.add(delta.address)
            counts[delta.address] += 1
            last[delta.address] = delta
    out: list[CursorCandidate] = []
    for address, hits in counts.most_common():
        if hits < min_hits:
            continue
        sample = last[address]
        out.append(
            CursorCandidate(
                address=address,
                axis=axis,
                hits=hits,
                last_before=sample.before,
                last_after=sample.after,
            )
        )
    return out


def stable_bytes(
    before: np.ndarray,
    after: np.ndarray,
    *,
    limit: int = 64,
) -> list[int]:
    """Return addresses that stayed equal (useful as anti-candidates)."""
    equal = np.flatnonzero(before == after)
    return [int(a) for a in equal[:limit]]


def intersect_changed(
    groups: list[list[RamDelta]],
) -> list[int]:
    """Addresses that changed in every group (scene-id style candidates)."""
    if not groups:
        return []
    sets = [{d.address for d in group} for group in groups]
    common = sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0]
    return sorted(common)


def ram_copy(ram: np.ndarray) -> np.ndarray:
    """Alias for snapshot() kept for call-site clarity."""
    return snapshot(ram)


def deltas_for_move(
    before: np.ndarray,
    after: np.ndarray,
    *,
    limit: int | None = 512,
) -> list[RamDelta]:
    """Diff helper with a Waldo-oriented default limit."""
    return diff_changed(before, after, limit=limit)
