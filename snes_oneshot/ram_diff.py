"""Differential RAM helpers for discovery workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RamDelta:
    """One address that changed between two RAM snapshots."""

    address: int
    before: int
    after: int

    @property
    def delta(self) -> int:
        """Signed change after - before."""
        return self.after - self.before


def snapshot(ram: np.ndarray) -> np.ndarray:
    """Copy a RAM buffer for later comparison."""
    return np.array(ram, dtype=np.uint8, copy=True)


def diff_changed(
    before: np.ndarray,
    after: np.ndarray,
    *,
    limit: int | None = 256,
) -> list[RamDelta]:
    """Return addresses whose byte values changed.

    Args:
        before: Earlier RAM snapshot.
        after: Later RAM snapshot.
        limit: Optional max number of deltas to return (sorted by address).
    """
    if before.shape != after.shape:
        raise ValueError("RAM snapshots must have the same shape")
    changed = np.flatnonzero(before != after)
    deltas = [
        RamDelta(address=int(addr), before=int(before[addr]), after=int(after[addr]))
        for addr in changed
    ]
    if limit is not None:
        return deltas[:limit]
    return deltas


def candidates_increasing(deltas: list[RamDelta]) -> list[RamDelta]:
    """Filter deltas that increased (useful for X/score probes)."""
    return [d for d in deltas if d.delta > 0]


def candidates_decreasing(deltas: list[RamDelta]) -> list[RamDelta]:
    """Filter deltas that decreased (useful for health probes)."""
    return [d for d in deltas if d.delta < 0]
