"""Tests for speedrun curriculum presets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speedrun_curriculum import CURRICULUM_TIERS, get_liukang_tiers


@pytest.mark.parametrize("name", sorted(CURRICULUM_TIERS))
def test_tier_weights_sum_to_one(name: str) -> None:
    tiers = get_liukang_tiers(name)
    total = sum(weight for _, weight, _ in tiers)
    assert total == pytest.approx(1.0)


def test_unknown_curriculum_raises() -> None:
    with pytest.raises(ValueError, match="Unknown curriculum"):
        get_liukang_tiers("invalid")
