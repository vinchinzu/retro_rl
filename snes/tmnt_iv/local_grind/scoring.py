"""Scalar score for keep/discard (lower is better)."""

from __future__ import annotations

from typing import Any, Mapping

# Prefer clear outcomes; punish life loss / timeout hard.
_OUTCOME_PENALTY: dict[str, float] = {
    "cleared": 0.0,
    "stage_advance": 0.0,
    "boss_down": 500.0,
    "timeout": 50_000.0,
    "life_loss": 100_000.0,
    "forbidden_a": 100_000.0,
}


def score_metrics(metrics: Mapping[str, Any]) -> float:
    """Lower is better: frames + 40×damage + 200×heals + outcome penalty."""
    frames = float(metrics.get("frames", 0) or 0)
    damage = float(metrics.get("damage_taken", 0) or 0)
    heals = float(metrics.get("heals", 0) or 0)
    outcome = str(metrics.get("outcome", "timeout"))
    penalty = _OUTCOME_PENALTY.get(outcome, 25_000.0)
    return frames + 40.0 * damage + 200.0 * heals + penalty


def is_improvement(
    candidate: float,
    baseline: float,
    *,
    min_rel_gain: float = 0.01,
) -> bool:
    """True when candidate beats baseline by at least ``min_rel_gain``."""
    if baseline <= 0:
        return candidate < baseline
    return candidate <= baseline * (1.0 - min_rel_gain)


__all__ = ["is_improvement", "score_metrics"]
