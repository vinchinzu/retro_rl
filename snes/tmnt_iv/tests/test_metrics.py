"""KEEP/discard scoring: finish first, then lower frames and damage."""

from __future__ import annotations

from tmnt_iv.local_grind.scoring import is_improvement, score_metrics


def _score(*, outcome: str, frames: int, damage: int, heals: int = 0) -> float:
    return score_metrics(
        {"outcome": outcome, "frames": frames, "damage_taken": damage, "heals": heals}
    )


def test_score_prefers_finish_then_time_then_damage() -> None:
    clear = _score(outcome="cleared", frames=10_000, damage=400, heals=5)
    faster = _score(outcome="cleared", frames=9_000, damage=400, heals=5)
    safer = _score(outcome="cleared", frames=10_000, damage=300, heals=5)
    timeout = _score(outcome="timeout", frames=10_000, damage=400, heals=5)
    death = _score(outcome="life_loss", frames=10_000, damage=400, heals=5)
    special = _score(outcome="forbidden_a", frames=10_000, damage=0, heals=0)

    assert faster < clear < timeout < death
    assert safer < clear
    assert special > timeout
    assert is_improvement(clear * 0.95, clear)
    assert not is_improvement(clear, clear)
