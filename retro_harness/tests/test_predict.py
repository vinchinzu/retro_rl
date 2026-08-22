"""Shared claim grammar (no emulator)."""

from __future__ import annotations

import pytest

from retro_harness.predict import (
    MissingPrediction,
    first_miss_index,
    grade_claims,
    parse_claims,
)


def test_refuse_empty_prediction() -> None:
    with pytest.raises(MissingPrediction):
        parse_claims("")
    with pytest.raises(MissingPrediction):
        parse_claims("   ")


def test_move_and_screen_grade() -> None:
    before = {"x": 120, "y": 141, "screen": 0x7C, "mode": 5}
    hit = {"x": 119, "y": 141, "screen": 0x7C, "mode": 5}
    grade = grade_claims("move -1,0; screen=0x7c", before, hit)
    assert grade.ok
    assert grade.missed == ()

    stuck = {"x": 120, "y": 141, "screen": 0x7C, "mode": 5}
    miss = grade_claims("move -1,0", before, stuck)
    assert not miss.ok
    assert miss.missed == ("move -1,0",)


def test_approx_and_screen() -> None:
    after = {"x": 118, "y": 93, "screen": 0x6B}
    grade = grade_claims("x≈120±4; screen=0x6b", {"x": 120, "y": 100}, after)
    assert grade.ok

    default_tol = grade_claims("x≈120", {"x": 120}, {"x": 118})
    assert default_tol.ok


def test_noop_change_and_first_miss() -> None:
    pose = {"x": 1, "y": 2, "mode": 5}
    assert grade_claims("noop", pose, pose).ok
    moved = {"x": 2, "y": 2, "mode": 5}
    assert grade_claims("change", pose, moved).ok
    assert not grade_claims("noop", pose, moved).ok
    grades = [
        grade_claims("x=1", {}, {"x": 1}),
        grade_claims("x=2", {}, {"x": 9}),
        grade_claims("x=3", {}, {"x": 3}),
    ]
    assert first_miss_index(grades) == 1


def test_unknown_claim_misses_even_if_pose_changed() -> None:
    before = {"x": 1, "y": 2, "mode": 5}
    after = {"x": 9, "y": 2, "mode": 5}
    grade = grade_claims("something changes", before, after)
    assert not grade.ok
    assert grade.missed == ("something changes",)


def test_unrecognized_move_and_bare_words_miss() -> None:
    before = {"x": 1, "y": 2}
    after = {"x": 2, "y": 2}
    typo = grade_claims("mov 1,0", before, after)
    assert not typo.ok
    assert typo.missed == ("mov 1,0",)
    bare = grade_claims("walk left", before, after)
    assert not bare.ok
    assert bare.missed == ("walk left",)
