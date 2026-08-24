"""Unit tests for the wiki-charge Phantoon bench helpers (no emulator)."""

from __future__ import annotations

import pytest

from super_metroid.combat.phantoon import WEAPON_BEAM
from super_metroid.scripts.probe.phantoon_wiki_charge import (
    POLICIES,
    group_window_chips,
    make_strategy,
    summarize_window_chips,
)


def test_policies_match_probe_and_spine() -> None:
    assert POLICIES["probe_default"] == 1
    assert POLICIES["spine_three"] == 3
    one = make_strategy("probe_default", 40_000)
    three = make_strategy("spine_three", 12_000)
    assert one.weapon == WEAPON_BEAM
    assert three.weapon == WEAPON_BEAM
    assert one.shots_per_window == 1
    assert three.shots_per_window == 3
    assert one.max_fight_frames == 40_000
    assert three.max_fight_frames == 12_000


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        make_strategy("super_spray", 100)


def test_group_window_chips_splits_on_gap() -> None:
    chips = [
        {"frame": 100, "drop": 300, "hp_after": 2200},
        {"frame": 130, "drop": 0, "hp_after": 2200},
        {"frame": 2500, "drop": 300, "hp_after": 1900},
        {"frame": 2580, "drop": 100, "hp_after": 1800},
        {"frame": 9000, "drop": 300, "hp_after": 1500},
    ]
    groups = group_window_chips(chips, gap=240)
    assert [len(g) for g in groups] == [1, 2, 1]
    assert groups[1][1]["drop"] == 100


def test_summarize_disappear_after_300() -> None:
    ones = [[{"frame": i * 1000, "drop": 300}] for i in range(8)]
    ones.append([{"frame": 9000, "drop": 100}])
    summary = summarize_window_chips(ones)
    assert summary["window_count"] == 9
    assert summary["max"] == 1
    assert summary["three_chips_landed"] is False
    assert summary["disappear_after_300"] is True

    triple = [[{"frame": 10, "drop": 300}, {"frame": 80, "drop": 300}, {"frame": 150, "drop": 300}]]
    triple_summary = summarize_window_chips(triple)
    assert triple_summary["three_chips_landed"] is True
    assert triple_summary["disappear_after_300"] is False
