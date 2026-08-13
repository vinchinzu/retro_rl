"""Unit tests for shared hillclimb frame-save policy."""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.platformer.frame_save import (
    accept_candidate,
    resolve_frame_save_mode,
)


@dataclass
class _Result:
    completed: bool = False
    total_frames: int = 0
    fitness: float = 0.0


def test_resolve_none_auto_from_completed_seed() -> None:
    mode = resolve_frame_save_mode(True, None, None)
    assert mode.prefer_trim is True
    assert mode.require_completion is True


def test_resolve_none_auto_from_incomplete_seed() -> None:
    mode = resolve_frame_save_mode(False, None, None)
    assert mode.prefer_trim is False
    assert mode.require_completion is False


def test_resolve_explicit_overrides() -> None:
    mode = resolve_frame_save_mode(True, prefer_trim=False, require_completion=False)
    assert mode.prefer_trim is False
    assert mode.require_completion is False
    mode = resolve_frame_save_mode(False, prefer_trim=True, require_completion=True)
    assert mode.prefer_trim is True
    assert mode.require_completion is True


def test_accept_require_completion_prefers_fewer_frames() -> None:
    best = _Result(completed=True, total_frames=100, fitness=900.0)
    better_frames = _Result(completed=True, total_frames=98, fitness=800.0)
    worse_frames = _Result(completed=True, total_frames=101, fitness=999.0)
    assert accept_candidate(best, better_frames, require_completion=True)
    assert not accept_candidate(best, worse_frames, require_completion=True)


def test_accept_require_completion_fitness_tiebreak() -> None:
    best = _Result(completed=True, total_frames=100, fitness=900.0)
    tie_better = _Result(completed=True, total_frames=100, fitness=901.0)
    tie_worse = _Result(completed=True, total_frames=100, fitness=899.0)
    assert accept_candidate(best, tie_better, require_completion=True)
    assert not accept_candidate(best, tie_worse, require_completion=True)


def test_accept_rejects_incomplete_when_required() -> None:
    best = _Result(completed=True, total_frames=100, fitness=900.0)
    incomplete = _Result(completed=False, total_frames=50, fitness=9999.0)
    assert not accept_candidate(best, incomplete, require_completion=True)


def test_accept_fitness_only_without_gating() -> None:
    best = _Result(completed=False, total_frames=100, fitness=100.0)
    better = _Result(completed=False, total_frames=200, fitness=150.0)
    worse = _Result(completed=True, total_frames=50, fitness=90.0)
    assert accept_candidate(best, better, require_completion=False)
    assert not accept_candidate(best, worse, require_completion=False)


def test_accept_first_completion_when_best_incomplete() -> None:
    best = _Result(completed=False, total_frames=200, fitness=50.0)
    clear = _Result(completed=True, total_frames=180, fitness=10.0)
    assert accept_candidate(best, clear, require_completion=True)
