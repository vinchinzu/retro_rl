"""Unit tests for TMNT II Stage1Policy (no emulator)."""

from __future__ import annotations

from tmnt_ii.policy import Stage1Policy


def test_open_phase_uses_right_and_attack() -> None:
    policy = Stage1Policy(target_score=5)
    reasons = {policy.tick(frame=f, score=0, health=60).reason for f in range(1, 30)}
    assert "open_walk" in reasons
    assert "open_attack" in reasons or "open_rb" in reasons


def test_lock_phase_faces_left() -> None:
    policy = Stage1Policy(target_score=5)
    reasons = {policy.tick(frame=f, score=3, health=60).reason for f in range(1, 25)}
    assert "lock_face" in reasons
    assert any(r.startswith("lock_") for r in reasons)


def test_push_phase_after_target() -> None:
    policy = Stage1Policy(target_score=5)
    reasons = {policy.tick(frame=f, score=5, health=50).reason for f in range(1, 35)}
    assert "push_walk" in reasons
    assert "push_attack" in reasons


def test_dead_idles() -> None:
    policy = Stage1Policy()
    tick = policy.tick(frame=1, score=0, health=0)
    assert tick.reason == "dead"


def test_play_relative_frame_one_opens_walk() -> None:
    """M4 leftover hands the policy play-relative frame 1, not boot time."""
    policy = Stage1Policy(target_score=5)
    tick = policy.tick(frame=1, score=0, health=60)
    assert tick.reason == "open_walk"
