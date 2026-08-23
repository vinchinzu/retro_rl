"""ROM-free tests for the RAM-scripted Liu Kang policy."""

from __future__ import annotations

import numpy as np

from mortal_kombat.ram import make_test_ram
from mortal_kombat.scripted import (
    B,
    DOWN,
    KIND_SCRIPT,
    L,
    LEFT,
    RIGHT,
    X,
    Y,
    ScriptedPolicy,
    fireball_sequence,
    flying_kick_sequence,
)

RGB = None


def _bits(frame: np.ndarray) -> set[int]:
    return {i for i, v in enumerate(frame) if v}


def test_kind_script_constant() -> None:
    assert KIND_SCRIPT == "script"
    assert ScriptedPolicy.kind == "script"
    assert ScriptedPolicy.name == "scripted"


def test_fireball_facing_and_hp() -> None:
    right = fireball_sequence(1)
    left = fireball_sequence(-1)
    assert len(right) == 15
    assert len(left) == 15
    assert right[0][RIGHT] == 1
    assert right[0][LEFT] == 0
    assert left[0][LEFT] == 1
    assert left[0][RIGHT] == 0
    for frame in right[-4:]:
        assert frame[Y] == 1
        assert frame[B] == 0
    for frame in right[:-4]:
        assert frame[Y] == 0


def test_flying_kick_uses_hk_not_hp() -> None:
    seq = flying_kick_sequence(1)
    assert len(seq) == 15
    assert seq[0][RIGHT] == 1
    for frame in seq[-4:]:
        assert frame[B] == 1
        assert frame[Y] == 0
    for frame in seq[:-4]:
        assert frame[B] == 0


def _policy() -> ScriptedPolicy:
    return ScriptedPolicy(intro_frames=0)


def test_far_apart_enqueues_fireball() -> None:
    policy = _policy()
    facing_right = policy.act(make_test_ram(p1_x=40, p2_x=200), RGB)
    assert facing_right[RIGHT] == 1
    assert facing_right[LEFT] == 0
    assert facing_right[Y] == 0

    policy.reset()
    facing_left = policy.act(make_test_ram(p1_x=200, p2_x=40), RGB)
    assert facing_left[LEFT] == 1
    assert facing_left[RIGHT] == 0


def test_close_walks_back_not_fireball() -> None:
    policy = _policy()
    frame = policy.act(make_test_ram(p1_x=100, p2_x=140), RGB)
    assert _bits(frame) == {LEFT}
    assert frame[Y] == 0
    assert frame[B] == 0


def test_airborne_opponent_uppercut() -> None:
    policy = _policy()
    frame = policy.act(make_test_ram(p1_x=80, p2_x=120, p2_y=100), RGB)
    assert frame[DOWN] == 1
    assert frame[L] == 1
    assert frame[Y] == 0
    assert frame[X] == 0


def test_p2_attacking_close_blocks() -> None:
    policy = _policy()
    ram = make_test_ram(p1_x=80, p2_x=120, p2_state=1)
    frame = policy.act(ram, RGB)
    assert frame[X] == 1
    assert frame[L] == 0
    assert frame[Y] == 0


def test_non_fight_screen_noop() -> None:
    policy = _policy()
    frame = policy.act(make_test_ram(p1_health=0, p2_health=0, timer=0), RGB)
    assert frame.shape == (12,)
    assert frame.dtype == np.int8
    assert int(frame.sum()) == 0


def test_act_drains_queue_without_redeciding() -> None:
    policy = _policy()
    far = make_test_ram(p1_x=40, p2_x=200)
    expected = fireball_sequence(1)
    close = make_test_ram(p1_x=100, p2_x=110)
    frames = [policy.act(far, RGB)]
    frames.extend(policy.act(close, RGB) for _ in range(14))
    assert len(frames) == 15
    for got, want in zip(frames, expected, strict=True):
        assert np.array_equal(got, want)
    after = policy.act(close, RGB)
    assert _bits(after) == {LEFT}


def test_reset_clears_queue() -> None:
    policy = _policy()
    policy.act(make_test_ram(p1_x=40, p2_x=200), RGB)
    policy.reset()
    frame = policy.act(make_test_ram(p1_x=100, p2_x=140), RGB)
    assert _bits(frame) == {LEFT}


def test_mid_range_walks_back_to_zone() -> None:
    policy = _policy()
    frame = policy.act(make_test_ram(p1_x=80, p2_x=140), RGB)
    assert _bits(frame) == {LEFT}
    assert frame[Y] == 0
    assert frame[B] == 0


def test_cooldown_blocks_immediate_refire() -> None:
    policy = _policy()
    ram = make_test_ram(p1_x=40, p2_x=200)
    for _ in range(15):
        policy.act(ram, RGB)
    follow = [policy.act(ram, RGB) for _ in range(15)]
    assert all(frame[Y] == 0 for frame in follow)
    assert all(int(frame.sum()) == 0 for frame in follow)


def test_intro_hold_is_noop_then_fireball() -> None:
    policy = ScriptedPolicy(intro_frames=3)
    ram = make_test_ram(p1_x=40, p2_x=200)
    for _ in range(3):
        frame = policy.act(ram, RGB)
        assert int(frame.sum()) == 0
    start = policy.act(ram, RGB)
    assert start[RIGHT] == 1
    assert start[Y] == 0
