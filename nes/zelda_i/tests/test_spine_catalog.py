"""Catalog invariants for Survival --through ids, stops, and LeaveSpecs."""

from __future__ import annotations

# Import the assembled spine first: it primes door_graph before level5.spine,
# which otherwise trips a circular import when imported in isolation.
from zelda_i.spine.survival import BOOT_POLICY, SPINE_THROUGH, SpineRun
from zelda_i.level5.spine import L5_STOPS, L5_THROUGH
from zelda_i.level6.spine import L6_STOPS, L6_THROUGH
from zelda_i.level7.spine import L7_STOPS, L7_THROUGH
from zelda_i.level8.spine import L8_STOPS, L8_THROUGH
from zelda_i.screen_glance import (
    BOW22_LEAVE,
    BOW_CELLAR_LEAVE,
    BOW_PICKUP_LEAVE,
    CELLAR08_LEAVE,
    CLEAR_3A,
    GOHMA_LEAVE,
    HEART_LEAVE,
    LEVEL6_LEAVE,
    NORTH0C_LEAVE,
    NORTH2C_LEAVE,
    SOUTH1D_LEAVE,
    STAIRS3A_DEST,
    WEST2D_LEAVE,
)

LEAVE_SPECS = (
    CLEAR_3A,
    CELLAR08_LEAVE,
    SOUTH1D_LEAVE,
    WEST2D_LEAVE,
    NORTH2C_LEAVE,
    GOHMA_LEAVE,
    HEART_LEAVE,
    NORTH0C_LEAVE,
    LEVEL6_LEAVE,
    BOW22_LEAVE,
    BOW_CELLAR_LEAVE,
    BOW_PICKUP_LEAVE,
    STAIRS3A_DEST,
)


def test_spine_through_unique_nonempty_and_suffixes() -> None:
    assert SPINE_THROUGH
    assert len(SPINE_THROUGH) == len(set(SPINE_THROUGH))
    assert L5_THROUGH and L6_THROUGH and L7_THROUGH and L8_THROUGH
    suffix = L5_THROUGH + L6_THROUGH + L7_THROUGH + L8_THROUGH
    prefix_len = len(SPINE_THROUGH) - len(suffix)
    prefix = SPINE_THROUGH[:prefix_len]
    assert prefix and prefix[0] == "level1" and prefix[-1] == "level4"
    assert SPINE_THROUGH == prefix + suffix
    assert SPINE_THROUGH[-len(L8_THROUGH) :] == L8_THROUGH
    l7_start = SPINE_THROUGH.index(L7_THROUGH[0])
    assert SPINE_THROUGH[l7_start : l7_start + len(L7_THROUGH)] == L7_THROUGH
    start = SPINE_THROUGH.index(L5_THROUGH[0])
    assert SPINE_THROUGH[start : start + len(L5_THROUGH)] == L5_THROUGH
    l6_start = SPINE_THROUGH.index(L6_THROUGH[0])
    assert SPINE_THROUGH[l6_start : l6_start + len(L6_THROUGH)] == L6_THROUGH


def test_l5_l6_stops_keys_match_through() -> None:
    assert set(L6_STOPS) == set(L6_THROUGH)
    assert set(L5_STOPS) == set(L5_THROUGH)
    assert set(L7_STOPS) == set(L7_THROUGH)
    assert set(L8_STOPS) == set(L8_THROUGH)


def test_leave_spec_hops_unique_and_on_spine() -> None:
    hops = [spec.hop for spec in LEAVE_SPECS]
    assert hops
    assert len(hops) == len(set(hops))
    assert all(hop in SPINE_THROUGH for hop in hops)


def test_spine_run_gohma_report_stop() -> None:
    run = SpineRun(through="level6-gohma", success=True, boot_frames=199)
    assert run.report()["stop"] == L6_STOPS["level6-gohma"]


def test_spine_run_level6_report_stop() -> None:
    run = SpineRun(through="level6", success=True, boot_frames=199)
    assert run.report()["stop"] == L6_STOPS["level6"]


def test_boot_policy_file_slot_and_quest() -> None:
    assert BOOT_POLICY["file_slot"] == 1 and BOOT_POLICY["quest"] == 1
