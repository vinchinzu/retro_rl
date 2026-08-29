"""Catalog invariants for Survival --through ids, stops, and LeaveSpecs."""

from __future__ import annotations

from pathlib import Path

import pytest

import zelda_i.survival_spine as spine
from zelda_i.level5_spine import L5_STOPS, L5_THROUGH
from zelda_i.level6_spine import L6_STOPS, L6_THROUGH
from zelda_i.screen_glance import (
    BOW22_LEAVE,
    BOW_CELLAR_LEAVE,
    BOW_PICKUP_LEAVE,
    CELLAR08_LEAVE,
    CLEAR_3A,
    NORTH2C_LEAVE,
    SOUTH1D_LEAVE,
    STAIRS3A_DEST,
    WEST2D_LEAVE,
)
from zelda_i.survival_spine import BOOT_POLICY, SPINE_THROUGH, SpineRun

LEAVE_SPECS = (
    CLEAR_3A,
    CELLAR08_LEAVE,
    SOUTH1D_LEAVE,
    WEST2D_LEAVE,
    NORTH2C_LEAVE,
    BOW22_LEAVE,
    BOW_CELLAR_LEAVE,
    BOW_PICKUP_LEAVE,
    STAIRS3A_DEST,
)


def test_spine_through_unique_nonempty_and_suffixes() -> None:
    assert SPINE_THROUGH
    assert len(SPINE_THROUGH) == len(set(SPINE_THROUGH))
    assert L5_THROUGH and L6_THROUGH
    prefix_len = len(SPINE_THROUGH) - len(L5_THROUGH) - len(L6_THROUGH)
    prefix = SPINE_THROUGH[:prefix_len]
    assert prefix and prefix[0] == "level1" and prefix[-1] == "level4"
    assert SPINE_THROUGH == prefix + L5_THROUGH + L6_THROUGH
    assert SPINE_THROUGH[-len(L6_THROUGH) :] == L6_THROUGH
    start = SPINE_THROUGH.index(L5_THROUGH[0])
    assert SPINE_THROUGH[start : start + len(L5_THROUGH)] == L5_THROUGH


def test_l5_l6_stops_keys_match_through() -> None:
    assert set(L6_STOPS) == set(L6_THROUGH)
    assert set(L5_STOPS) == set(L5_THROUGH)


def test_leave_spec_hops_unique_and_on_spine() -> None:
    hops = [spec.hop for spec in LEAVE_SPECS]
    assert hops
    assert len(hops) == len(set(hops))
    assert all(hop in SPINE_THROUGH for hop in hops)


def test_spine_run_gohma_report_stop() -> None:
    run = SpineRun(through="level6-gohma", success=True, boot_frames=199)
    assert run.report()["stop"] == L6_STOPS["level6-gohma"]


def test_boot_policy_file_slot_and_quest() -> None:
    assert BOOT_POLICY["file_slot"] == 1 and BOOT_POLICY["quest"] == 1


def test_seamed_compose_module_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("zelda_i.scripts.compose_honest_route_recording")
    names = [name.lower() for name in dir(spine)]
    assert not any("compose" in name or "seam" in name for name in names)
    root = Path(__file__).resolve().parents[1]
    gone = (
        root / "compose_survival.py",
        root / "scripts" / "compose_honest_route_recording.py",
        root / "scripts" / "compose_survival.py",
    )
    assert not any(path.exists() for path in gone)
