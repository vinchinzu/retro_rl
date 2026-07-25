"""Tests for train_speedrun curriculum tier presets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MK_DIR))

from speedrun_curriculum import CURRICULUM_TIERS, get_liukang_tiers


def _tier_weight_sum(tiers: list[tuple[list[str], float, str]]) -> float:
    return sum(weight for _, weight, _ in tiers)


@pytest.mark.parametrize("curriculum", sorted(CURRICULUM_TIERS))
def test_curriculum_weights_sum_to_one(curriculum: str) -> None:
    tiers = get_liukang_tiers(curriculum)
    assert _tier_weight_sum(tiers) == pytest.approx(1.0)


def test_ladder_curriculum_emphasizes_mid_ladder() -> None:
    full = {name: weight for _, weight, name in get_liukang_tiers("full")}
    ladder = {name: weight for _, weight, name in get_liukang_tiers("ladder")}

    assert ladder["Medium (M4-M6)"] > full["Medium (M4-M6)"]
    assert ladder["Goro (sub-boss)"] < full["Goro (sub-boss)"]
    assert ladder["Shang Tsung (final)"] < full["Shang Tsung (final)"]


def test_unknown_curriculum_raises() -> None:
    with pytest.raises(ValueError, match="Unknown curriculum"):
        get_liukang_tiers("invalid")
