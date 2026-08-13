"""Shared frame-save policy for hillclimb accept/mutation bias.

Used by index hillclimb, raw hillclimb, and segment hillclimb so completion
gating and trim bias stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ScoredResult(Protocol):
    """Minimal result shape needed for accept decisions."""

    completed: bool
    total_frames: int
    fitness: float


@dataclass(frozen=True, slots=True)
class FrameSaveMode:
    """Resolved frame-save policy after applying seed-based defaults."""

    prefer_trim: bool
    require_completion: bool


def resolve_frame_save_mode(
    seed_completed: bool,
    prefer_trim: bool | None,
    require_completion: bool | None,
) -> FrameSaveMode:
    """Resolve None flags from whether the seed already finishes.

    ``None`` means auto: finishing seeds enable frame-save mode (trim bias +
    completion gating); incomplete seeds leave both off.
    """
    return FrameSaveMode(
        prefer_trim=seed_completed if prefer_trim is None else prefer_trim,
        require_completion=(
            seed_completed if require_completion is None else require_completion
        ),
    )


def accept_candidate(
    best_result: ScoredResult,
    cand_result: ScoredResult,
    *,
    require_completion: bool,
) -> bool:
    """Whether *cand_result* should replace *best_result*.

    With completion gating: must complete, then prefer fewer ``total_frames``,
    then higher fitness on a frame tie. Without gating: fitness only.
    """
    if require_completion:
        if not cand_result.completed:
            return False
        if not best_result.completed:
            return True
        if cand_result.total_frames < best_result.total_frames:
            return True
        if (
            cand_result.total_frames == best_result.total_frames
            and cand_result.fitness > best_result.fitness
        ):
            return True
        return False
    return cand_result.fitness > best_result.fitness


# Mutation strategy tables keyed by prefer_trim.
INDEX_MUTATION_WEIGHTS: dict[bool, tuple[list[str], list[int]]] = {
    True: (["single", "delete", "swap", "shift", "run_change"], [20, 35, 10, 15, 20]),
    False: (["single", "delete", "swap", "shift", "run_change"], [45, 5, 15, 15, 20]),
}

RAW_MUTATION_WEIGHTS: dict[bool, tuple[list[str], list[int]]] = {
    True: (["toggle", "delete", "shift_edge", "copy_run", "insert"], [15, 40, 20, 10, 15]),
    False: (["toggle", "delete", "shift_edge", "copy_run", "insert"], [30, 20, 20, 15, 15]),
}

SEGMENT_MUTATION_WEIGHTS: dict[bool, dict[str, int]] = {
    True: {
        "delete": 40,
        "trim_hold": 25,
        "shift_edge": 15,
        "toggle": 10,
        "copy_run": 5,
        "insert": 5,
    },
    False: {
        "toggle": 30,
        "delete": 20,
        "shift_edge": 20,
        "copy_run": 15,
        "insert": 15,
        "trim_hold": 0,
    },
}
